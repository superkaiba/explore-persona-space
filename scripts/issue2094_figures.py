"""Issue #2094 P10 — hero + exploratory figure dump (plan section 6).

Reads unit E's ``eval_results/issue_2094/`` outputs (f-tables, transport,
linearity, fragility, bootstrap CIs, best-cells) plus unit D's judge work root
(audits, optional stage-2 scores) and writes every plan-section-6 figure to
``figures/issue_2094/`` as PNG + PDF + ``.meta.json`` (provenance + per-point
data via ``savefig_paper``).

Conventions (one color = one meaning across the whole set):

* settings  — matched-prefix / matched-query / cross each keep ONE palette
  color everywhere (``SETTING_COLORS``);
* arms      — steered = the "primary" role color, shuffled-donor null = the
  "baseline" role color, everywhere;
* <50-percent-coherent cells are OVERLAID with a visible x marker — never
  grayed out or suppressed (plan section 4.5); cells with NO coherent draw at
  all show no value and carry the same marker.

Figures (stems): hero1_f_act_heatmap, hero1_f_act_dose_curves,
hero2_f_beh_heatmap, hero2_f_beh_dose_response, result1b_transport_cosines,
result1c_homogeneity, result1c_l_fit, result1c_operator_2x2,
result2c_transfer_decomposition, result3_fact_vs_fbeh,
result3_fbeh_vs_traversal, result4_fragility, exp_anchor_separation,
exp_fact_layer_profiles, exp_typeA_vs_typeB, exp_query_prefix_marginals,
exp_stage2_vs_stage1, exp_audit_rates.

Usage (VM; thread caps per the shared-VM rule):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue2094_figures.py \\
        --out-root eval_results/issue_2094 --fig-dir figures/issue_2094
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue2094_figures")

# ── shared conventions ─────────────────────────────────────────────────

SETTING_ORDER: tuple[str, ...] = ("matched_prefix", "matched_query", "cross")
SETTING_LABELS: dict[str, str] = {
    "matched_prefix": "matched prefix (query transfer)",
    "matched_query": "matched query (prefix transfer)",
    "cross": "cross (joint transfer)",
}
DOSE_ORDER: tuple[str, ...] = ("a0.5", "a1", "a2", "a4", "replace")
DOSE_LABELS: dict[str, str] = {
    "a0.5": "dose 0.5x",
    "a1": "dose 1x",
    "a2": "dose 2x",
    "a4": "dose 4x",
    "replace": "replace (full state)",
}
SLOT_ORDER: tuple[str, ...] = ("ce", "pe", "cm2", "cm3", "l3j", "qspan")
SLOT_LABELS: dict[str, str] = {
    "ce": "context end",
    "pe": "prefix end",
    "cm2": "2nd-to-last token",
    "cm3": "3rd-to-last token",
    "l3j": "last-3 joint",
    "qspan": "query span",
}
JOINT_LABELS: dict[str, str] = {"joint_mid": "joint mid-stack", "joint_all": "joint all-layer"}
KIND_LABELS: dict[str, str] = {"query": "query rubric", "prefix": "prefix rubric"}

LOW_COHERENCE_FRAC = 0.5  # cells below this coherent fraction get the overlay marker

_SETTING_COLORS: dict[str, str] = dict(zip(SETTING_ORDER, paper_palette(3), strict=True))
STEERED_COLOR = paper_palette_role("primary")
NULL_COLOR = paper_palette_role("baseline")


def setting_color(setting: str) -> str:
    return _SETTING_COLORS[setting]


# ── input loading ──────────────────────────────────────────────────────


def _iter_jsonl(path: Path):
    if not path.exists():
        # A leg-specific out-root (e.g. the transport leg) may not carry every
        # table; the REQUIRED figures still fail loud with a named assertion.
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class FigInputs:
    """Everything the figure builders read, loaded once."""

    f_cells: list[dict] = field(default_factory=list)
    null_cells: list[dict] = field(default_factory=list)
    anchors: list[dict] = field(default_factory=list)
    anchor_draws: list[dict] = field(default_factory=list)
    bootstrap: dict | None = None
    transport_cells: list[dict] = field(default_factory=list)
    l_fit: dict | None = None
    operator_cmp: dict | None = None
    homogeneity: dict | None = None
    fragility: dict | None = None
    best_cells: dict | None = None
    stage2_scores: list[dict] = field(default_factory=list)
    audit_rows: dict[str, list[dict]] = field(default_factory=dict)


def load_inputs(out_root: Path, judge_root: Path) -> FigInputs:
    fm = out_root / "f_metrics"
    inp = FigInputs(
        f_cells=list(_iter_jsonl(fm / "f_cells.jsonl")),
        null_cells=list(_iter_jsonl(fm / "null_cells.jsonl")),
        anchors=list(_iter_jsonl(fm / "anchors.jsonl")),
        anchor_draws=list(_iter_jsonl(fm / "anchor_draws.jsonl")),
        bootstrap=_load_json(fm / "bootstrap_cis.json"),
        l_fit=_load_json(out_root / "linearity" / "l_fit_results.json"),
        operator_cmp=_load_json(out_root / "linearity" / "operator_comparison.json"),
        homogeneity=_load_json(out_root / "linearity" / "homogeneity.json"),
        fragility=_load_json(out_root / "fragility" / "fragility_cells.json"),
        best_cells=_load_json(out_root / "best_cells.json"),
    )
    tcells = out_root / "transport" / "transport_cells.jsonl"
    if tcells.exists():
        inp.transport_cells = list(_iter_jsonl(tcells))
    scores_dir = judge_root / "scores"
    if scores_dir.is_dir():
        for f in sorted(scores_dir.glob("*.scores.jsonl")):
            for row in _iter_jsonl(f):
                if row.get("kind") == "stage2":
                    inp.stage2_scores.append(row)
    audits_dir = judge_root / "audits"
    if audits_dir.is_dir():
        for f in sorted(audits_dir.glob("*.audit.jsonl")):
            inp.audit_rows[f.stem.removesuffix(".audit")] = list(_iter_jsonl(f))
    return inp


# ── row helpers ────────────────────────────────────────────────────────


def row_f_beh(row: dict) -> float | None:
    """Mean F_beh over the row's available rubric kinds (None when all missing)."""
    vals = [
        v["f_beh"]
        for v in (row.get("f_beh") or {}).values()
        if isinstance(v, dict) and v.get("f_beh") is not None
    ]
    return float(np.mean(vals)) if vals else None


def row_metric(row: dict, metric: str) -> float | None:
    if metric == "f_act":
        v = row.get("f_act")
        return None if v is None else float(v)
    assert metric == "f_beh", metric
    return row_f_beh(row)


def dose_alpha(dose: str) -> float | None:
    return None if dose == "replace" else float(dose.removeprefix("a"))


def _lv_sort_key(lv: str):
    if lv.startswith("L") and lv[1:].isdigit():
        return (0, int(lv[1:]))
    return (1, list(JOINT_LABELS).index(lv) if lv in JOINT_LABELS else 99)


def lv_label(lv: str) -> str:
    if lv.startswith("L") and lv[1:].isdigit():
        return lv[1:]
    return JOINT_LABELS.get(lv, lv)


@dataclass
class CellAgg:
    """One (setting, dose, slot, layer-variant) cell family aggregated over pairs."""

    mean: float | None
    n_values: int
    n_rows: int
    coherent_frac: float


def aggregate_cells(rows: list[dict], metric: str) -> dict[tuple[str, str, str, str], CellAgg]:
    """(setting, dose, slot, layer_variant) -> CellAgg over pairs."""
    acc: dict[tuple[str, str, str, str], dict] = {}
    for row in rows:
        key = (row["setting"], row["dose"], row["slot"], row["layer_variant"])
        rec = acc.setdefault(key, {"vals": [], "n_rows": 0, "n_coh": 0})
        rec["n_rows"] += 1
        rec["n_coh"] += int(bool(row.get("coherent")))
        v = row_metric(row, metric)
        if v is not None and math.isfinite(v):
            rec["vals"].append(v)
    out = {}
    for key, rec in acc.items():
        out[key] = CellAgg(
            mean=float(np.mean(rec["vals"])) if rec["vals"] else None,
            n_values=len(rec["vals"]),
            n_rows=rec["n_rows"],
            coherent_frac=rec["n_coh"] / rec["n_rows"] if rec["n_rows"] else 0.0,
        )
    return out


def overlay_low_coherence(ax, xs: list[float], ys: list[float]) -> None:
    """VISIBLE marker on low-coherence cells (never grayed / suppressed)."""
    if xs:
        ax.scatter(
            xs,
            ys,
            marker="x",
            s=45,
            linewidths=1.6,
            color="black",
            zorder=5,
            label="<50 percent coherent",
        )


# ── HERO 1 / HERO 2: F heatmaps ────────────────────────────────────────


def fig_f_heatmap(rows: list[dict], metric: str, title: str) -> plt.Figure:
    """Slot x layer heatmap grid, setting columns x dose rows; joint variants as
    extra marked columns; low-coherence cells overlaid with x markers."""
    agg = aggregate_cells(rows, metric)
    doses = [d for d in DOSE_ORDER if any(k[1] == d for k in agg)]
    settings = [s for s in SETTING_ORDER if any(k[0] == s for k in agg)]
    lvs = sorted({k[3] for k in agg}, key=_lv_sort_key)
    slots = [s for s in SLOT_ORDER if any(k[2] == s for k in agg)]
    n_single = sum(1 for lv in lvs if lv.startswith("L") and lv[1:].isdigit())
    assert doses and settings and lvs and slots, "no aggregable cells"

    fig, axes = plt.subplots(
        len(doses),
        len(settings),
        figsize=(3.4 * len(settings) + 1.6, 1.9 * len(doses) + 1.2),
        squeeze=False,
        layout="constrained",
    )
    finite = [c.mean for c in agg.values() if c.mean is not None]
    vmax = max(1.0, float(np.percentile(np.abs(finite), 98))) if finite else 1.0
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")
    im = None
    for i, dose in enumerate(doses):
        for j, setting in enumerate(settings):
            ax = axes[i][j]
            mat = np.full((len(slots), len(lvs)), np.nan)
            low_x, low_y = [], []
            for (st, do, sl, lv), cell in agg.items():
                if st != setting or do != dose:
                    continue
                y, x = slots.index(sl), lvs.index(lv)
                if cell.mean is not None:
                    mat[y, x] = cell.mean
                if cell.coherent_frac < LOW_COHERENCE_FRAC:
                    low_x.append(x)
                    low_y.append(y)
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                aspect="auto",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            overlay_low_coherence(ax, low_x, low_y)
            if n_single and n_single < len(lvs):
                ax.axvline(n_single - 0.5, color="black", linewidth=1.0)
            ax.set_xticks(range(len(lvs)))
            ax.set_xticklabels([lv_label(lv) for lv in lvs], rotation=90, fontsize=6)
            ax.set_yticks(range(len(slots)))
            ax.set_yticklabels(
                [SLOT_LABELS[s] for s in slots] if j == 0 else [""] * len(slots), fontsize=7
            )
            ax.grid(False)
            if i == 0:
                ax.set_title(SETTING_LABELS[setting], fontsize=8)
            if j == 0:
                ax.set_ylabel(DOSE_LABELS[dose], fontsize=8)
    fig.supxlabel("steered layer (right of divider: joint variants)")
    fig.colorbar(im, ax=axes, shrink=0.8, label=title)
    fig.suptitle(f"{title} — mean over pairs; x marker = <50 percent coherent cell")
    return fig


# ── HERO dose curves / dose-response ───────────────────────────────────


def _family_ci(
    bootstrap: dict | None,
    arm: str,
    setting: str,
    slot: str,
    lv: str,
    dose: str,
    vec_type: str,
    metric_key: str,
) -> dict | None:
    if not bootstrap:
        return None
    fams = bootstrap.get("families", {})
    return fams.get("|".join([arm, setting, slot, lv, dose, vec_type, metric_key]))


def pair_slopes(by_pair: dict[str, dict[float, float]]) -> dict[str, float]:
    """Per-pair OLS slope of F vs log2(alpha) (pairs with >=2 doses)."""
    out = {}
    for pid, d in by_pair.items():
        if len(d) < 2:
            continue
        xs = np.log2(np.array(sorted(d)))
        ys = np.array([d[a] for a in sorted(d)])
        out[pid] = float(np.polyfit(xs, ys, 1)[0])
    return out


def fig_dose_response(
    steered: list[dict],
    metric: str,
    title: str,
    bootstrap: dict | None,
    layer_variant: str | None = None,
) -> plt.Figure:
    """Per-pair spaghetti vs log2-dose + bootstrap mean band + donor-null band,
    plus the per-pair slope distribution with a signed-rank read in the title."""
    # per-SLOT most-populated single-layer variant (a slot with no additive
    # single-layer rows contributes no row — production sweeps every layer on
    # ce/pe, but a partial run / smoke may not).
    lv_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in steered:
        if r["dose"] != "replace" and r["layer_variant"].startswith("L"):
            lv_counts[r["slot"]][r["layer_variant"]] += 1
    assert lv_counts, "no additive single-layer cells"
    lv_by_slot = {
        slot: (layer_variant or max(c, key=lambda k: c[k])) for slot, c in lv_counts.items()
    }
    slots = [s for s in ("ce", "pe") if s in lv_by_slot] or sorted(lv_by_slot)[:1]
    settings = [s for s in SETTING_ORDER if any(r["setting"] == s for r in steered)]

    fig, axes = plt.subplots(
        len(slots) + 1,
        len(settings),
        figsize=(3.2 * len(settings) + 0.6, 2.3 * (len(slots) + 1)),
        squeeze=False,
        layout="constrained",
    )
    all_slopes: list[float] = []
    for i, slot in enumerate(slots):
        lv = lv_by_slot[slot]
        for j, setting in enumerate(settings):
            ax = axes[i][j]
            by_pair: dict[str, dict[float, float]] = defaultdict(dict)
            for r in steered:
                if r["slot"] != slot or r["setting"] != setting or r["layer_variant"] != lv:
                    continue
                a, v = dose_alpha(r["dose"]), row_metric(r, metric)
                if a is not None and v is not None:
                    by_pair[r["pair_id"]][a] = v
            for pid, d in sorted(by_pair.items()):
                xs = sorted(d)
                ax.plot(
                    np.log2(xs),
                    [d[a] for a in xs],
                    color=setting_color(setting),
                    alpha=0.35,
                    linewidth=0.9,
                    marker="o",
                    markersize=2.2,
                )
            mkey = "f_act" if metric == "f_act" else None
            for arm, color, label in (
                ("steered", STEERED_COLOR, "steered mean (bootstrap 95 percent)"),
                ("null", NULL_COLOR, "shuffled-donor null"),
            ):
                xs_c, mid, lo, hi = [], [], [], []
                for dose in ("a0.5", "a1", "a2", "a4"):
                    keys = [mkey] if mkey else ["f_beh_query", "f_beh_prefix"]
                    cis = [
                        c
                        for k in keys
                        if k
                        for c in [_family_ci(bootstrap, arm, setting, slot, lv, dose, "A", k)]
                        if c and c.get("observed_mean") is not None
                    ]
                    if not cis:
                        continue
                    xs_c.append(math.log2(dose_alpha(dose)))
                    mid.append(float(np.mean([c["observed_mean"] for c in cis])))
                    lo.append(
                        float(
                            np.mean([c["ci_lo"] for c in cis if c["ci_lo"] is not None] or [np.nan])
                        )
                    )
                    hi.append(
                        float(
                            np.mean([c["ci_hi"] for c in cis if c["ci_hi"] is not None] or [np.nan])
                        )
                    )
                if xs_c:
                    # markers keep a single-dose cell visible (a 1-point line draws nothing)
                    ax.plot(
                        xs_c,
                        mid,
                        color=color,
                        linewidth=2.0,
                        label=label,
                        marker="o",
                        markersize=3.5,
                    )
                    if np.isfinite(lo).all() and np.isfinite(hi).all():
                        ax.fill_between(xs_c, lo, hi, color=color, alpha=0.25, linewidth=0)
            ax.axhline(0.0, color="grey", linewidth=0.6)
            ax.set_xlabel("log2 dose")
            if j == 0:
                ax.set_ylabel(f"{title}\n{SLOT_LABELS[slot]} @ {lv}")
            if i == 0:
                ax.set_title(SETTING_LABELS[setting], fontsize=8)
            if i == 0 and j == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6)
            slopes = pair_slopes(by_pair)
            all_slopes.extend(slopes.values())
    # slope distribution row (pooled over slots, per setting)
    for j, setting in enumerate(settings):
        ax = axes[len(slots)][j]
        by_pair_all: dict[str, dict[float, float]] = defaultdict(dict)
        for r in steered:
            if r["setting"] != setting or r["layer_variant"] != lv_by_slot.get(r["slot"]):
                continue
            a, v = dose_alpha(r["dose"]), row_metric(r, metric)
            if a is not None and v is not None:
                by_pair_all[(r["pair_id"], r["slot"])][a] = v
        slopes = list(pair_slopes(by_pair_all).values())
        sr = ""
        if len(slopes) >= 3 and any(s != 0 for s in slopes):
            from scipy.stats import wilcoxon

            try:
                res = wilcoxon(slopes)
                sr = f" — signed-rank p={res.pvalue:.3g} (n={len(slopes)})"
            except ValueError:
                sr = f" — signed-rank undefined (n={len(slopes)})"
        if slopes:
            ax.hist(slopes, bins=15, color=setting_color(setting))
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"per-pair slope{sr}", fontsize=7)
        ax.set_xlabel("slope of F vs log2 dose")
        if j == 0:
            ax.set_ylabel("pairs")
    lv_note = ", ".join(f"{SLOT_LABELS[s]} @ {lv_by_slot[s]}" for s in slots)
    fig.suptitle(f"{title} dose response ({lv_note}; additive doses)")
    return fig


# ── Result 1b: transport cosines ───────────────────────────────────────


def fig_transport(transport_cells: list[dict]) -> plt.Figure:
    assert transport_cells, "no transport rows"
    groups = sorted({r["map_id"] for r in transport_cells})
    dose_classes = [
        dc
        for dc in ("additive", "replace")
        if any((r["dose"] == "replace") == (dc == "replace") for r in transport_cells)
    ]
    fig, axes = plt.subplots(
        1,
        len(dose_classes),
        figsize=(1.1 * len(groups) * len(dose_classes) + 3.0, 3.6),
        squeeze=False,
        layout="constrained",
    )
    rng = np.random.default_rng(0)
    for d_i, dc in enumerate(dose_classes):
        ax = axes[0][d_i]
        for g_i, map_id in enumerate(groups):
            for a_i, (arm, color) in enumerate((("steered", STEERED_COLOR), ("null", NULL_COLOR))):
                vals = [
                    r["cosine_tail"]
                    for r in transport_cells
                    if r["map_id"] == map_id
                    and r["arm"] == arm
                    and ((r["dose"] == "replace") == (dc == "replace"))
                    and r["cosine_tail"] is not None
                ]
                x0 = g_i + (a_i - 0.5) * 0.35
                if vals:
                    ax.bar(
                        x0,
                        float(np.mean(vals)),
                        width=0.3,
                        color=color,
                        alpha=0.8,
                        label=(
                            {"steered": "steered", "null": "shuffled-donor null"}[arm]
                            if g_i == 0 and d_i == 0
                            else None
                        ),
                    )
                    ax.scatter(
                        x0 + rng.uniform(-0.06, 0.06, len(vals)),
                        vals,
                        s=8,
                        color="black",
                        alpha=0.5,
                        zorder=4,
                    )
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(
            [
                g.replace("m779_ce_", "context map ").replace("m1738_pe_", "prefix map ")
                for g in groups
            ],
            rotation=30,
            ha="right",
            fontsize=7,
        )
        ax.set_ylabel("cos(realized shift, map-predicted shift)")
        ax.set_title("additive doses (pooled)" if dc == "additive" else "replace (full state)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Banked-map transport at context-end / prefix-end cells")
    return fig


# ── Result 1c: homogeneity + L fit + 2x2 ───────────────────────────────


def fig_homogeneity(homog: dict) -> plt.Figure:
    fams = {k: v for k, v in (homog.get("families") or {}).items() if v}
    assert fams, "no homogeneity families"
    fig, axes = plt.subplots(
        len(fams), 2, figsize=(7.4, 2.6 * len(fams)), squeeze=False, layout="constrained"
    )
    for i, (fam, pairs) in enumerate(sorted(fams.items())):
        mats, alphas_ref = [], None
        ax_m, ax_n = axes[i][0], axes[i][1]
        for pid, rec in sorted(pairs.items()):
            alphas = rec["alphas"]
            if alphas_ref is None or len(alphas) > len(alphas_ref):
                alphas_ref = alphas
            m = np.array(rec["disattenuated_cosine_matrix"], dtype=float)
            if mats and m.shape != mats[0].shape:
                continue
            mats.append(m)
            norms = np.array(rec["shift_norms"], dtype=float)
            ax_n.plot(alphas, norms, marker="o", markersize=3, alpha=0.5, linewidth=0.9)
        mean_mat = np.nanmean(np.stack(mats), axis=0) if mats else np.zeros((1, 1))
        im = ax_m.imshow(mean_mat, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax_m.set_xticks(range(len(alphas_ref)))
        ax_m.set_xticklabels([f"{a:g}x" for a in alphas_ref], fontsize=7)
        ax_m.set_yticks(range(len(alphas_ref)))
        ax_m.set_yticklabels([f"{a:g}x" for a in alphas_ref], fontsize=7)
        ax_m.grid(False)
        ax_m.set_title(f"{fam}: mean disattenuated cos(shift@dose_i, shift@dose_j)", fontsize=7)
        fig.colorbar(im, ax=ax_m, shrink=0.8)
        # unity-slope reference through the median alpha=1 norm
        a_arr = np.array(alphas_ref, dtype=float)
        ones = [
            np.array(rec["shift_norms"], dtype=float)[rec["alphas"].index(1.0)]
            for rec in pairs.values()
            if 1.0 in rec["alphas"]
        ]
        if ones:
            ref = float(np.median(ones))
            ax_n.plot(
                a_arr,
                ref * a_arr,
                linestyle="--",
                color="black",
                linewidth=1.0,
                label="unity slope",
            )
            ax_n.legend(fontsize=6)
        ax_n.set_xscale("log", base=2)
        ax_n.set_yscale("log", base=2)
        ax_n.set_xlabel("dose (log scale)")
        ax_n.set_ylabel("realized shift norm")
        ax_n.set_title(f"{fam}: norm vs dose per pair (log-log)", fontsize=7)
    fig.suptitle("Linearity: dose-homogeneity of realized shifts")
    return fig


def fig_l_fit(l_fit: dict) -> plt.Figure:
    fams = l_fit.get("families") or {}
    assert fams, "no L-fit families"
    keys = sorted(fams)
    fig, axes = plt.subplots(2, 1, figsize=(1.5 * len(keys) + 4.0, 6.4), layout="constrained")
    ax = axes[0]
    bar_specs = (
        ("pair-fold OOF R2", lambda r: (r.get("pair_fold") or {}).get("pooled_r2")),
        ("held-out-family OOF R2", lambda r: (r.get("family_fold") or {}).get("pooled_r2")),
        ("identity+bias baseline R2", lambda r: r.get("identity_bias_pooled_oof_r2")),
    )
    colors = paper_palette(len(bar_specs))
    for b_i, (label, getter) in enumerate(bar_specs):
        xs, ys = [], []
        for k_i, key in enumerate(keys):
            v = getter(fams[key])
            if v is not None:
                xs.append(k_i + (b_i - 1) * 0.27)
                ys.append(float(v))
        ax.bar(xs, ys, width=0.25, color=colors[b_i], label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("pooled held-out R2")
    ax.set_title("One-operator fit L: dose-delta to realized shift (PC-128 ridge)", fontsize=9)
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ks = (1, 5, 10)
    width = 0.8 / (len(ks) * 2)
    for k_i, key in enumerate(keys):
        knn = fams[key].get("knn_retrieval") or {}
        for m_i, metric in enumerate(("euclidean", "cosine")):
            rec = knn.get(metric)
            if not rec:
                continue
            accs = rec.get("acc_at_k") or {}
            chance = rec.get("chance_at_k") or {}
            for kk_i, kk in enumerate(ks):
                acc = accs.get(str(kk), accs.get(kk))
                if acc is None:
                    continue
                x = k_i + (m_i * len(ks) + kk_i - 2.5) * width
                ax2.bar(
                    x,
                    float(acc),
                    width=width * 0.9,
                    color=paper_palette(2)[m_i],
                    alpha=0.4 + 0.2 * kk_i,
                    label=(f"{metric} acc@k" if k_i == 0 and kk_i == 0 else None),
                )
                ch = chance.get(str(kk), chance.get(kk))
                if ch is not None:
                    ax2.plot([x - width / 2, x + width / 2], [ch, ch], color="black", linewidth=0.9)
    ax2.set_xticks(range(len(keys)))
    ax2.set_xticklabels(keys, rotation=20, ha="right", fontsize=7)
    ax2.set_ylabel("retrieval accuracy at k in 1, 5, 10")
    ax2.set_title("kNN retrieval of the true realized shift (black line = chance)", fontsize=9)
    ax2.legend(fontsize=7)
    return fig


def fig_operator_2x2(operator_cmp: dict) -> plt.Figure:
    comps = operator_cmp.get("comparisons") or {}
    two = operator_cmp.get("two_by_two") or {}
    assert comps or two, "no operator comparisons"
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
    ax = axes[0]
    keys = sorted(comps)
    for i, key in enumerate(keys):
        c = comps[key]
        ax.bar(
            i - 0.15,
            c["procrustes_cosine_subspace"],
            width=0.28,
            color=STEERED_COLOR,
            label="Procrustes operator cosine" if i == 0 else None,
        )
        ax.bar(
            i + 0.15,
            c["raw_cosine_subspace"],
            width=0.28,
            color=NULL_COLOR,
            label="raw operator cosine" if i == 0 else None,
        )
        ax.plot(
            [i - 0.32, i + 0.02], [c["procrustes_null_p97_5"]] * 2, color="black", linewidth=1.2
        )
        ax.plot(
            [i - 0.02, i + 0.32],
            [c["raw_null_p97_5_abs"]] * 2,
            color="black",
            linewidth=1.2,
            linestyle="--",
        )
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(
        [k.replace("_vs_", " vs ") for k in keys], rotation=10, ha="right", fontsize=7
    )
    ax.set_ylabel("operator cosine")
    ax.set_title("Fitted L vs banked maps (lines = matched rotation-null p97.5)", fontsize=8)
    ax.legend(fontsize=7)

    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for key, rec in sorted(two.items()):
        rows.append(
            [
                key,
                "yes" if rec["M_aligns"] else "no",
                "yes" if rec["L_predicts"] else "no",
                f"{rec['procrustes_cosine']:.3f}",
                f"{rec['procrustes_null_p97_5']:.3f}",
                "n/a" if rec["family_fold_r2"] is None else f"{rec['family_fold_r2']:.3f}",
            ]
        )
    table = ax2.table(
        cellText=rows or [["(none)"] * 6],
        colLabels=[
            "family",
            "map aligns?",
            "L predicts?",
            "Procrustes cos",
            "null p97.5",
            "family-fold R2",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    ax2.set_title("2x2: does the banked map align x does a linear L predict", fontsize=8)
    return fig


# ── Result 2c / 3 ──────────────────────────────────────────────────────


def fig_transfer_decomposition(rows: list[dict]) -> plt.Figure:
    pts = []
    for r in rows:
        if r["setting"] != "cross":
            continue
        beh = r.get("f_beh") or {}
        fp = (beh.get("prefix") or {}).get("f_beh")
        fq = (beh.get("query") or {}).get("f_beh")
        if fp is not None and fq is not None:
            pts.append((fp, fq, r["dose"]))
    assert pts, "no cross-setting cells with both rubric kinds"
    fig, ax = plt.subplots(figsize=(4.6, 4.2), layout="constrained")
    doses = sorted({p[2] for p in pts}, key=lambda d: DOSE_ORDER.index(d))
    colors = dict(zip(doses, paper_palette(max(len(doses), 2)), strict=False))
    for dose in doses:
        sel = [(x, y) for x, y, d in pts if d == dose]
        ax.scatter(
            *zip(*sel, strict=True), s=12, alpha=0.6, color=colors[dose], label=DOSE_LABELS[dose]
        )
    lim = max(abs(v) for p in pts for v in p[:2]) * 1.05 + 1e-9
    ax.plot([-lim, lim], [-lim, lim], color="grey", linewidth=0.6, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("F on the prefix rubric (register transfer)")
    ax.set_ylabel("F on the query rubric (query transfer)")
    ax.set_title("Cross setting: joint-transfer decomposition per cell")
    ax.legend(fontsize=7)
    return fig


def fig_fact_vs_fbeh(rows: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    n = 0
    for setting in SETTING_ORDER:
        xs, ys = [], []
        for r in rows:
            if r["setting"] != setting:
                continue
            fa, fb = r.get("f_act"), row_f_beh(r)
            if fa is not None and fb is not None:
                xs.append(fa)
                ys.append(fb)
        if xs:
            n += len(xs)
            ax.scatter(
                xs,
                ys,
                s=10,
                alpha=0.55,
                color=setting_color(setting),
                label=SETTING_LABELS[setting],
            )
    assert n, "no cells with both F levels"
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("F_act (activation-level fraction of swap)")
    ax.set_ylabel("F_beh (behavior-level fraction of swap)")
    ax.set_title("Activation vs behavior movement per steered cell")
    ax.legend(fontsize=7)
    return fig


def fig_fbeh_vs_traversal(rows: list[dict]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    n = 0
    for setting in SETTING_ORDER:
        xs, ys = [], []
        for r in rows:
            if r["setting"] != setting:
                continue
            tr, fb = r.get("traversal_ratio"), row_f_beh(r)
            if tr is not None and fb is not None:
                xs.append(tr)
                ys.append(fb)
        if xs:
            n += len(xs)
            ax.scatter(
                xs,
                ys,
                s=10,
                alpha=0.55,
                color=setting_color(setting),
                label=SETTING_LABELS[setting],
            )
    assert n, "no cells with traversal + F_beh"
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_xlabel("realized traversal (shift norm / anchor-axis norm)")
    ax.set_ylabel("F_beh")
    ax.set_title("Behavior movement vs realized activation traversal")
    ax.legend(fontsize=7)
    return fig


# ── Result 4: fragility ────────────────────────────────────────────────


def fig_fragility(fragility: dict) -> plt.Figure:
    cells = fragility.get("cells") or []
    assert cells, "no fragility cells"
    doses = [d for d in DOSE_ORDER if any(c["dose"] == d for c in cells)]
    lvs = sorted({c["layer_variant"] for c in cells}, key=_lv_sort_key)
    slots = [s for s in SLOT_ORDER if any(c["slot"] == s for c in cells)]
    panels = (
        ("steered", "excess_incoherence", "steered excess incoherence"),
        ("null", "excess_incoherence", "shuffled-donor excess incoherence"),
        ("steered", "cap_hit_frac", "steered cap-hit fraction (companion)"),
    )
    fig, axes = plt.subplots(
        len(doses),
        len(panels),
        figsize=(3.2 * len(panels) + 1.4, 1.8 * len(doses) + 1.0),
        squeeze=False,
        layout="constrained",
    )
    ims = []
    for i, dose in enumerate(doses):
        for j, (arm, field_name, label) in enumerate(panels):
            ax = axes[i][j]
            mat = np.full((len(slots), len(lvs)), np.nan)
            for c in cells:
                if c["dose"] != dose:
                    continue
                v = c[arm].get(field_name)
                if v is not None:
                    mat[slots.index(c["slot"]), lvs.index(c["layer_variant"])] = v
            cmap = plt.get_cmap("magma_r").copy()
            cmap.set_bad("#E8E8E8")
            vmax = (
                1.0
                if field_name == "cap_hit_frac"
                else max(0.2, float(np.nanmax(mat)) if np.isfinite(mat).any() else 0.2)
            )
            vmin = (
                0.0
                if field_name == "cap_hit_frac"
                else min(0.0, float(np.nanmin(mat)) if np.isfinite(mat).any() else 0.0)
            )
            ims.append(
                ax.imshow(
                    np.ma.masked_invalid(mat),
                    aspect="auto",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
            )
            ax.set_xticks(range(len(lvs)))
            ax.set_xticklabels([lv_label(lv) for lv in lvs], rotation=90, fontsize=6)
            ax.set_yticks(range(len(slots)))
            ax.set_yticklabels(
                [SLOT_LABELS[s] for s in slots] if j == 0 else [""] * len(slots), fontsize=7
            )
            ax.grid(False)
            if i == 0:
                ax.set_title(label, fontsize=8)
            if j == 0:
                ax.set_ylabel(DOSE_LABELS[dose], fontsize=8)
            fig.colorbar(ims[-1], ax=ax, shrink=0.75)
    base = fragility.get("anchor_baseline") or {}
    fig.suptitle(
        "Fragility: incoherence beyond the anchor baseline "
        f"(baseline incoherent frac = {base.get('incoherent_frac', float('nan')):.3f}); "
        "cap-hit counted next to, never blended with, incoherence"
    )
    return fig


# ── exploratory dump ───────────────────────────────────────────────────


def fig_anchor_separation(anchors: list[dict]) -> plt.Figure:
    assert anchors, "no anchor pair stats"
    settings = [s for s in SETTING_ORDER if any(a["setting"] == s for a in anchors)]
    fig, axes = plt.subplots(
        1,
        len(settings),
        figsize=(3.4 * len(settings) + 0.8, 3.4),
        squeeze=False,
        layout="constrained",
    )
    for j, setting in enumerate(settings):
        ax = axes[0][j]
        rows = sorted(
            (a for a in anchors if a["setting"] == setting),
            key=lambda a: (a["pair_id"], a["kind"]),
        )
        labels, seps, colors = [], [], []
        kind_colors = dict(zip(("query", "prefix"), paper_palette(2), strict=True))
        for a in rows:
            labels.append(f"{a['pair_id']} ({KIND_LABELS[a['kind']]})")
            seps.append(a["separation"] if a["separation"] is not None else np.nan)
            colors.append(kind_colors[a["kind"]])
        ax.barh(range(len(rows)), seps, color=colors)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=5)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("ceiling minus floor anchor contrast")
        ax.set_title(SETTING_LABELS[setting], fontsize=8)
    fig.suptitle("Anchor separation per pair (the F denominator)")
    return fig


def fig_fact_layer_profiles(rows: list[dict]) -> plt.Figure:
    by_group: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for r in rows:
        prof = r.get("f_act_profile")
        if prof and not r.get("excluded_incoherent"):
            by_group[(r["slot"], r["dose"])].append([np.nan if v is None else v for v in prof])
    assert by_group, "no f_act profiles"
    slots = [s for s in SLOT_ORDER if any(k[0] == s for k in by_group)]
    fig, axes = plt.subplots(
        1,
        len(slots),
        figsize=(3.0 * len(slots) + 0.8, 3.2),
        squeeze=False,
        sharey=True,
        layout="constrained",
    )
    dose_colors = dict(zip(DOSE_ORDER, paper_palette(len(DOSE_ORDER)), strict=True))
    for j, slot in enumerate(slots):
        ax = axes[0][j]
        for dose in DOSE_ORDER:
            profs = by_group.get((slot, dose))
            if not profs:
                continue
            mean = np.nanmean(np.array(profs, dtype=float), axis=0)
            ax.plot(range(len(mean)), mean, color=dose_colors[dose], label=DOSE_LABELS[dose])
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_title(SLOT_LABELS[slot], fontsize=8)
        ax.set_xlabel("read layer")
        if j == 0:
            ax.set_ylabel("mean F_act")
            ax.legend(fontsize=6)
    fig.suptitle("F_act read-layer profiles (mean over steered cells)")
    return fig


def fig_type_ab(rows: list[dict]) -> plt.Figure:
    by_cell: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        v = r.get("f_act")
        if v is not None:
            by_cell[(r["slot"], r["layer_variant"], r["dose"], r["pair_id"])][r["vec_type"]] = v
    pts = [(d["A"], d["B"]) for d in by_cell.values() if "A" in d and "B" in d]
    assert pts, "no shared Type-A/Type-B cells"
    fig, ax = plt.subplots(figsize=(4.4, 4.2), layout="constrained")
    ax.scatter(*zip(*pts, strict=True), s=14, alpha=0.6, color=STEERED_COLOR)
    lim = max(abs(v) for p in pts for v in p) * 1.05 + 1e-9
    ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--", linewidth=0.7)
    ax.set_xlabel("F_act with the pair-difference vector (Type A)")
    ax.set_ylabel("F_act with the query-averaged centroid (Type B)")
    ax.set_title("Pair-specific vs centroid direction on shared cells")
    return fig


def _ctx_parts(cid: str) -> tuple[str, str]:
    prefix, query = cid.split("__", 1)
    return prefix, query


def fig_marginals(rows: list[dict]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), layout="constrained")
    for ax, part_i, what in ((axes[0], 1, "target query"), (axes[1], 0, "target prefix")):
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"f_act": [], "f_beh": []})
        for r in rows:
            key = _ctx_parts(r["context_b"])[part_i]
            fa, fb = r.get("f_act"), row_f_beh(r)
            if fa is not None:
                acc[key]["f_act"].append(fa)
            if fb is not None:
                acc[key]["f_beh"].append(fb)
        keys = sorted(acc)
        assert keys, "no marginal groups"
        colors = paper_palette(2)
        for m_i, metric in enumerate(("f_act", "f_beh")):
            vals = [float(np.mean(acc[k][metric])) if acc[k][metric] else np.nan for k in keys]
            ax.bar(
                [i + (m_i - 0.5) * 0.36 for i in range(len(keys))],
                vals,
                width=0.34,
                color=colors[m_i],
                label=metric,
            )
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"mean F by {what}", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Per-query and per-prefix marginals (steered cells)")
    return fig


def stage2_cell_f(
    stage2_scores: list[dict], anchors: list[dict], coherence_threshold: float = 60.0
) -> dict[tuple[str, str], dict]:
    """Stage-2 re-measured F_beh per (cell, rubric kind), mirroring the
    analysis conventions: per-draw contrast on coherent draws, anchor
    floor/ceiling normalization from ``anchors.jsonl``."""
    anch = {(a["pair_id"], a["kind"]): a for a in anchors}
    coh: dict[tuple, float | None] = {}
    beh: dict[tuple, dict] = defaultdict(dict)
    for row in stage2_scores:
        key = (row.get("cell") or row.get("block_key"), row["pair_id"], row.get("draw"))
        if row["rubric_id"] == "coherence":
            coh[key] = row["score"]
        else:
            beh[key][(row["rubric_kind"], row["side"])] = row["score"]
    out: dict[tuple[str, str], dict] = {}
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    for key, sides in beh.items():
        c = coh.get(key)
        if c is None or float(c) <= coherence_threshold:
            continue
        cell, pair_id, _draw = key
        for kind in ("query", "prefix"):
            sa, sb = sides.get((kind, "a")), sides.get((kind, "b"))
            st = anch.get((pair_id, kind))
            if sa is None or sb is None or st is None or st["separation"] in (None, 0):
                continue
            delta = (float(sb) - float(sa)) / 100.0
            f = (delta - st["floor"]["mean"]) / (st["ceiling"]["mean"] - st["floor"]["mean"])
            deltas[(cell, kind)].append(f)
            pair_of[(cell, kind)].add(pair_id)
    for k, vals in deltas.items():
        out[k] = {"mean_f": float(np.mean(vals)), "n_draws": len(vals), "n_pairs": len(pair_of[k])}
    return out


def fig_stage2_vs_stage1(
    stage2_scores: list[dict], anchors: list[dict], best_cells: dict | None
) -> plt.Figure:
    s2 = stage2_cell_f(stage2_scores, anchors)
    assert s2, "no coherent stage-2 scored draws"
    fig, ax = plt.subplots(figsize=(4.8, 4.2), layout="constrained")
    stage1 = {}
    for cell in (best_cells or {}).get("cells", []):
        key = "|".join(
            [cell["setting"], cell["slot"], cell["layer_variant"], cell["dose"], cell["vec_type"]]
        )
        stage1[key] = cell["mean_f"]
    xs, ys, labels = [], [], []
    for (cell, kind), rec in sorted(s2.items()):
        x = stage1.get(cell)
        if x is None:
            continue
        xs.append(x)
        ys.append(rec["mean_f"])
        labels.append(f"{cell} ({KIND_LABELS[kind]})")
    if not xs:  # smoke path: synthesized best-cells key set differs — show s2 alone
        for i, ((cell, kind), rec) in enumerate(sorted(s2.items())):
            xs.append(float(i))
            ys.append(rec["mean_f"])
            labels.append(f"{cell} ({KIND_LABELS[kind]})")
        ax.scatter(xs, ys, s=22, color=STEERED_COLOR)
        ax.set_xlabel("stage-2 cell index (no matching stage-1 selection)")
    else:
        ax.scatter(xs, ys, s=22, color=STEERED_COLOR)
        lim = max(abs(v) for v in xs + ys) * 1.1 + 1e-9
        ax.plot([-lim, lim], [-lim, lim], color="grey", linestyle="--", linewidth=0.7)
        ax.set_xlabel("stage-1 selected-cell mean F (greedy)")
    for x, y, lab in zip(xs, ys, labels, strict=True):
        ax.annotate(lab, (x, y), fontsize=5, xytext=(3, 3), textcoords="offset points")
    ax.set_ylabel("stage-2 re-measured mean F_beh (temp 1.0)")
    ax.set_title("Stage-2 confirmation vs stage-1 selection (post-selection, labeled)")
    return fig


def fig_audit_rates(audit_rows: dict[str, list[dict]]) -> plt.Figure:
    assert audit_rows, "no audit files"
    flags = ("flag_empty", "flag_script_intrusion", "flag_repetition")
    flag_labels = ("empty output", "non-Latin script intrusion", "4-gram repetition")
    kinds = sorted(audit_rows)
    fig, ax = plt.subplots(figsize=(2.2 * len(kinds) + 3.0, 3.4), layout="constrained")
    colors = paper_palette(len(flags))
    for f_i, (flag, flabel) in enumerate(zip(flags, flag_labels, strict=True)):
        xs, ys = [], []
        for k_i, kind in enumerate(kinds):
            rows = audit_rows[kind]
            xs.append(k_i + (f_i - 1) * 0.27)
            ys.append(float(np.mean([bool(r.get(flag)) for r in rows])) if rows else 0.0)
        ax.bar(xs, ys, width=0.25, color=colors[f_i], label=flabel)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_ylabel("flagged fraction of rollouts")
    ax.set_title("Mechanical audit rates per rollout source")
    ax.legend(fontsize=7)
    return fig


# ── registry + main ────────────────────────────────────────────────────


def build_all(inp: FigInputs, only: set[str] | None = None) -> dict[str, plt.Figure | str]:
    """Build every figure; returns {stem: Figure | skip-reason-string}."""
    steered = inp.f_cells
    both = inp.f_cells + inp.null_cells
    producers = {
        "hero1_f_act_heatmap": lambda: fig_f_heatmap(
            steered, "f_act", "F_act (fraction of activation swap)"
        ),
        "hero1_f_act_dose_curves": lambda: fig_dose_response(
            steered, "f_act", "F_act", inp.bootstrap
        ),
        "hero2_f_beh_heatmap": lambda: fig_f_heatmap(
            steered, "f_beh", "F_beh (fraction of behavior swap)"
        ),
        "hero2_f_beh_dose_response": lambda: fig_dose_response(
            steered, "f_beh", "F_beh", inp.bootstrap
        ),
        "result1b_transport_cosines": lambda: fig_transport(inp.transport_cells),
        "result1c_homogeneity": lambda: fig_homogeneity(inp.homogeneity or {}),
        "result1c_l_fit": lambda: fig_l_fit(inp.l_fit or {}),
        "result1c_operator_2x2": lambda: fig_operator_2x2(inp.operator_cmp or {}),
        "result2c_transfer_decomposition": lambda: fig_transfer_decomposition(steered),
        "result3_fact_vs_fbeh": lambda: fig_fact_vs_fbeh(steered),
        "result3_fbeh_vs_traversal": lambda: fig_fbeh_vs_traversal(steered),
        "result4_fragility": lambda: fig_fragility(inp.fragility or {}),
        "exp_anchor_separation": lambda: fig_anchor_separation(inp.anchors),
        "exp_fact_layer_profiles": lambda: fig_fact_layer_profiles(steered),
        "exp_typeA_vs_typeB": lambda: fig_type_ab(both),
        "exp_query_prefix_marginals": lambda: fig_marginals(steered),
        "exp_stage2_vs_stage1": lambda: fig_stage2_vs_stage1(
            inp.stage2_scores, inp.anchors, inp.best_cells
        ),
        "exp_audit_rates": lambda: fig_audit_rates(inp.audit_rows),
    }
    optional = {
        "exp_stage2_vs_stage1",
        "exp_audit_rates",
        "result1b_transport_cosines",
        # operator comparisons exist only where parity + banked-dim fits ran
        # (production / the production-dim smoke leg)
        "result1c_operator_2x2",
    }
    out: dict[str, plt.Figure | str] = {}
    for stem, producer in producers.items():
        if only and stem not in only:
            continue
        try:
            out[stem] = producer()
        except AssertionError as exc:
            if stem in optional:
                out[stem] = f"skipped (optional input absent): {exc}"
                logger.warning("[figures] %s SKIPPED: %s", stem, exc)
            else:
                raise
    return out


def _import_check() -> None:
    """Resolve every deferred import this script reaches on its real paths."""
    from scipy.stats import wilcoxon  # noqa: F401

    print("[import-check] OK", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #2094 P10 figure dump (plan section 6; paper-plots conventions)."
    )
    ap.add_argument("--out-root", type=Path, default=Path("eval_results/issue_2094"))
    ap.add_argument(
        "--judge-root",
        type=Path,
        default=None,
        help="unit-D judge work root (default: <out-root>/judge)",
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2094"))
    ap.add_argument("--only", type=str, default=None, help="comma-separated figure stems")
    ap.add_argument("--style", choices=("blog", "neurips", "generic"), default="blog")
    ap.add_argument("--import-check", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.import_check:
        _import_check()
        return 0
    set_paper_style(args.style)
    judge_root = args.judge_root if args.judge_root is not None else args.out_root / "judge"
    inp = load_inputs(args.out_root, judge_root)
    logger.info(
        "[figures] loaded: %d steered / %d null cells, %d anchors, transport=%d, stage2=%d",
        len(inp.f_cells),
        len(inp.null_cells),
        len(inp.anchors),
        len(inp.transport_cells),
        len(inp.stage2_scores),
    )
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    figs = build_all(inp, only)
    n_saved = 0
    for stem, fig in figs.items():
        if isinstance(fig, str):
            continue
        paths = savefig_paper(fig, stem, dir=args.fig_dir)
        plt.close(fig)
        n_saved += 1
        logger.info("[figures] saved %s -> %s", stem, paths.get("png"))
    skipped = {k: v for k, v in figs.items() if isinstance(v, str)}
    logger.info(
        "[phase=figures_done] saved=%d skipped=%d %s", n_saved, len(skipped), sorted(skipped)
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
