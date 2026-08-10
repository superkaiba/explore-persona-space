#!/usr/bin/env python3
"""Issue #2215 — figures over the Phase C outputs (plan section 6).

Pure PRESENTATION layer: every plotted number is read verbatim off the
Phase C artifacts under ``results_dir`` (``dv1_context_shift.json``,
``dv2_answer_shift.json``, ``dv3_map_discrimination.json``,
``coupling.json``, ``null_bands.json`` + ``perpair/dv3_pairs.jsonl``) — NO
statistic is recomputed here (loading, joining, and ordering only; the
hierarchically-clustered heatmap ORDER is presentation-level, plan
section 4.3). Driven by the pod driver's Phase D
(``issue2215_run.render_figures``): production renders into the repo's
``figures/issue_2215/`` (then committed by ``commit_results_git``);
smoke/tiny render into an out-root twin (smoke outputs never touch
committed paths).

Figure registry (plan section 6): two HERO figures + the exploratory dump.
A figure whose inputs are absent or degraded (tiny DV3 skip, <3-cell H2
skip, missing per-pair rows) records a SKIP reason in the returned
manifest — never an empty render, never a silent pass.

Conventions (/paper-plots skill): ``set_paper_style("blog")``,
``savefig_paper`` provenance sidecars, ONE color-to-meaning assignment
across the whole set — Wong palette indices 0-4 for the five DV3 arms,
5-7 for the DV1/DV2 measure series, neutral gray for the span-mean twin,
light gray for every shuffled-pair / permutation null band — error bars
wherever a CI is persisted (magnitude RATIOS carry no persisted CI; the
consistency CIs live in the consistency figure), no interpretive overlays,
plain-English rendered labels (cell ids prettified underscore-to-space;
arm slugs mapped to the plan section-5 condition names). The plan-named
hero bar set is {779ce, 1738pe, 1738ce, identity+bias(ce)}; the pe
identity baseline appears in the per-layer / kNN / transfer views.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE numeric imports (shared-VM thread caps)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless pod-side render
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    add_direction_arrow,
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue2215.figures")

POOL_PRIMARY = "tail"
METRIC_PRIMARY = "cosine"

_P = paper_palette(8)  # Wong, colorblind-safe
ARM_ORDER = ("779ce", "1738pe", "1738ce", "idbias_ce", "idbias_pe")
HERO_ARMS = ("779ce", "1738pe", "1738ce", "idbias_ce")  # plan section 6 hero bar set
GOAL_ARMS = ("779ce", "1738pe")  # Goal-named arms (worst-to-best sort key)
ARM_COLORS = {a: _P[i] for i, a in enumerate(ARM_ORDER)}
ARM_LABELS = {
    "779ce": "single-turn context-end map",
    "1738pe": "multi-turn prefix-end map",
    "1738ce": "multi-turn context-end map",
    "idbias_ce": "identity + bias (context-end)",
    "idbias_pe": "identity + bias (prefix-end)",
}
# Measures never reuse the arm colors (indices 0-4). Wong's remaining
# yellow (#F0E442) is unreadable as a marker/line on the off-white blog
# canvas, so the third measure takes a dark-gold substitute instead.
MEASURE_COLORS = {"dv1_ce": _P[5], "dv1_pe": "#B8860B", "dv2_tail": _P[7], "dv2_span": "#8c8c8c"}
MEASURE_LABELS = {
    "dv1_ce": "context vector, context-end",
    "dv1_pe": "context vector, prefix-end",
    "dv2_tail": "answer vector, tail-inclusive mean",
    "dv2_span": "answer vector, span mean (excl. tail)",
}
NULL_COLOR = "#c4c4c4"
REF_COLOR = "#888888"

FigResult = tuple["plt.Figure | None", "str | None"]


# ── loading (no recomputation — reads Phase C artifacts verbatim) ──────


@dataclass
class Results:
    dv1: dict
    dv2: dict
    dv3: dict
    coupling: dict
    bands: dict
    dv3_pairs: list[dict] = field(default_factory=list)

    @property
    def dv3_ok(self) -> bool:
        return "per_config" in self.dv3

    @property
    def dv3_layer(self) -> int:
        assert self.dv3_ok
        return int(self.dv3["meta"]["primary_layer"])

    def dv1_primary_idx(self) -> int:
        return list(self.dv1["layers"]).index(self.dv1["meta"]["primary_layer"])

    def dv2_primary_idx(self) -> int:
        return list(self.dv2["layers"]).index(self.dv2["meta"]["primary_layer"])


def load_results(results_dir: Path) -> Results:
    def rd(name: str) -> dict:
        p = results_dir / f"{name}.json"
        assert p.exists(), f"{p} missing — run Phase C (--phase c) first"
        return json.loads(p.read_text())

    pairs_path = results_dir / "perpair" / "dv3_pairs.jsonl"
    dv3_pairs = (
        [json.loads(ln) for ln in pairs_path.read_text().splitlines() if ln.strip()]
        if pairs_path.exists()
        else []
    )
    return Results(
        dv1=rd("dv1_context_shift"),
        dv2=rd("dv2_answer_shift"),
        dv3=rd("dv3_map_discrimination"),
        coupling=rd("coupling"),
        bands=rd("null_bands"),
        dv3_pairs=dv3_pairs,
    )


# ── shared helpers ─────────────────────────────────────────────────────


def pretty(cell: str) -> str:
    """Plain-English rendered label for a bank cell id (section 3.5)."""
    return cell.replace("_", " ")


def arms_present(res: Results, want: tuple[str, ...] = ARM_ORDER) -> list[str]:
    layer = res.dv3_layer
    keys = res.dv3["per_config"]
    return [a for a in want if f"{a}|L{layer}|{POOL_PRIMARY}" in keys]


def _per_type_acc(res: Results, arm: str, cell: str) -> dict | None:
    """Registered-config per-type record (cosine, tail, primary layer) or None."""
    rec = res.dv3["per_config"][f"{arm}|L{res.dv3_layer}|{POOL_PRIMARY}"]["per_type"].get(cell)
    if not isinstance(rec, dict):
        return None
    m = rec.get(METRIC_PRIMARY)
    return m if isinstance(m, dict) and m.get("acc") is not None else None


def type_order_worst_to_best(res: Results) -> list[str]:
    """The Goal's ranking axis: mean registered 2AFC accuracy across the
    present Goal-named arms, ascending (worst-discriminated first). Cells
    with no accuracy (degenerate/excluded) sort last. Falls back to the
    DV1 context-end magnitude ratio when DV3 was skipped (tiny)."""
    cells = sorted(res.dv1["per_cell"])
    if res.dv3_ok:
        goal = [a for a in GOAL_ARMS if a in arms_present(res)] or arms_present(res)

        def key(cell: str) -> float:
            vals = [m["acc"] for a in goal if (m := _per_type_acc(res, a, cell)) is not None]
            return float(np.mean(vals)) if vals else float("inf")

        return sorted(cells, key=key)

    def key1(cell: str) -> float:
        r = res.dv1["per_cell"][cell]["ce"]["primary"].get("ratio")
        return float(r) if r is not None else float("inf")

    return sorted(cells, key=key1)


def _xerr(val: float, ci: list | None) -> np.ndarray | None:
    """matplotlib xerr wants DELTAS (2, N), never absolute CI bounds."""
    if not ci or val is None:
        return None
    lo, hi = float(ci[0]), float(ci[1])
    return np.array([[max(0.0, val - lo)], [max(0.0, hi - val)]])


def _pertype_height(n: int, per: float = 0.30, base: float = 1.8) -> float:
    return max(3.4, base + per * n)


def _label_points(ax: plt.Axes, xs, ys, names, fontsize: float = 5.5) -> None:
    for x, y, name in zip(xs, ys, names):
        ax.text(x, y, f" {name}", fontsize=fontsize, va="center", color="#444444")


# ── HERO 1: per-type paired 2AFC accuracy at the registered config ─────


def fig_hero1_per_type_2afc(res: Results) -> FigResult:
    if not res.dv3_ok:
        return None, f"DV3 unavailable: {res.dv3.get('skipped', 'no per_config')}"
    arms = [a for a in HERO_ARMS if a in arms_present(res)]
    if not arms:
        return None, "no hero arms present in dv3 per_config"
    order = type_order_worst_to_best(res)
    n, k = len(order), len(arms)
    fig, ax = plt.subplots(figsize=(7.5, _pertype_height(n, per=0.16 * k)))
    bar_h = 0.8 / k
    band_labeled = False
    for j, arm in enumerate(arms):
        ys, accs, errs, bands = [], [], [], []
        for i, cell in enumerate(order):
            m = _per_type_acc(res, arm, cell)
            if m is None:
                continue  # degenerate-at-pe / all-excluded — gap, labeled in the artifact
            ys.append(i + (j - (k - 1) / 2) * bar_h)
            accs.append(m["acc"])
            errs.append(m.get("acc_ci95_clustered"))
            bands.append(m.get("null_band"))
        if not ys:
            continue
        err = np.zeros((2, len(ys)))
        for t, (a, ci) in enumerate(zip(accs, errs)):
            d = _xerr(a, ci)
            if d is not None:
                err[:, t : t + 1] = d
        ax.barh(
            ys,
            accs,
            height=bar_h * 0.92,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
            zorder=3,
        )
        # CI whiskers drawn separately: barh's own xerr trips a numpy>=1.25
        # DeprecationWarning ("ndim > 0 to a scalar") on single-bar groups.
        ax.errorbar(accs, ys, xerr=err, fmt="none", ecolor="#333333", lw=0.7, zorder=4)
        for y, band in zip(ys, bands):
            if not band:
                continue
            ax.plot(
                band,
                [y, y],
                color=NULL_COLOR,
                lw=2.4,
                solid_capstyle="butt",
                zorder=5,  # above the bars: the band's full extent stays visible
                label="shuffled-pair null 95% band" if not band_labeled else None,
            )
            band_labeled = True
    ax.axvline(0.5, color=REF_COLOR, lw=0.8, ls="--", zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty(c) for c in order], fontsize=7)
    ax.invert_yaxis()  # worst-discriminated at the top
    ax.set_xlim(0, 1)
    ax.set_xlabel("paired two-alternative accuracy (cosine)")
    add_direction_arrow(ax, "x", "up")
    ax.set_title(
        f"Per-type paired discrimination accuracy at layer {res.dv3_layer} "
        "(tail-inclusive answer targets)",
        loc="left",
    )
    ax.legend(loc="lower right", fontsize=7)
    return fig, None


# ── HERO 2: per-type shift-magnitude ratio vs yardstick ────────────────


def fig_hero2_shift_ratio(res: Results) -> FigResult:
    order = type_order_worst_to_best(res)
    n = len(order)
    series: dict[str, dict[str, float]] = {"dv1_ce": {}, "dv1_pe": {}, "dv2_tail": {}}
    for cell in order:
        d1 = res.dv1["per_cell"][cell]
        r = d1["ce"]["primary"].get("ratio")
        if r is not None:
            series["dv1_ce"][cell] = float(r)
        pe = d1.get("pe", {})
        if not pe.get("degenerate_at_pe"):
            r = pe.get("primary", {}).get("ratio")
            if r is not None:
                series["dv1_pe"][cell] = float(r)
        d2 = res.dv2["per_cell"][cell][POOL_PRIMARY]
        r = d2["primary"].get("ratio")
        if r is not None:
            series["dv2_tail"][cell] = float(r)
    fig, ax = plt.subplots(figsize=(7.0, _pertype_height(n, per=0.24)))
    for i, cell in enumerate(order):  # thin pairing line per type
        xs = [series[s][cell] for s in series if cell in series[s] and series[s][cell] > 0]
        if len(xs) >= 2:
            ax.plot([min(xs), max(xs)], [i, i], color="#d9d9d9", lw=0.8, zorder=1)
    offsets = {"dv1_ce": -0.18, "dv1_pe": 0.0, "dv2_tail": 0.18}  # ties stay visible
    for name, vals in series.items():
        pts = [
            (i + offsets[name], vals[c]) for i, c in enumerate(order) if c in vals and vals[c] > 0
        ]
        if not pts:
            continue
        ys, xs = zip(*pts)
        ax.scatter(xs, ys, s=22, color=MEASURE_COLORS[name], label=MEASURE_LABELS[name], zorder=3)
    ax.axvline(1.0, color=REF_COLOR, lw=0.8, ls="--", zorder=2)
    ax.set_xscale("log")
    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty(c) for c in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("median value-pair shift / yardstick (log scale)")
    ax.set_title(
        "Per-type shift magnitude vs its yardstick (carrier shift for context "
        "vectors; split-half draw noise for answer vectors)",
        loc="left",
    )
    ax.legend(loc="best", fontsize=7)
    return fig, None


# ── exploratory dump (plan section 6) ──────────────────────────────────


def fig_margin_scatter_per_type(res: Results) -> FigResult:
    if not res.dv3_pairs:
        return None, "no dv3 per-pair rows (tiny run, or perpair/dv3_pairs.jsonl absent)"
    order = type_order_worst_to_best(res)
    row_idx = {c: i for i, c in enumerate(order)}
    arms = [a for a in ("779ce", "1738pe", "1738ce") if any(r["arm"] == a for r in res.dv3_pairs)]
    if not arms:
        return None, "no fitted-arm rows in dv3_pairs.jsonl"
    rng = np.random.default_rng(2215)  # presentation-only y jitter
    fig, axes = plt.subplots(
        1,
        len(arms),
        figsize=(3.2 * len(arms) + 1.2, _pertype_height(len(order), per=0.22)),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        xs, ys = [], []
        for r in res.dv3_pairs:
            if r["arm"] != arm or r["cell"] not in row_idx:
                continue
            for m in (r["margin_cos_a"], r["margin_cos_b"]):
                xs.append(m)
                ys.append(row_idx[r["cell"]] + float(rng.uniform(-0.28, 0.28)))
        ax.scatter(xs, ys, s=5, alpha=0.35, color=ARM_COLORS[arm], linewidths=0)
        ax.axvline(0.0, color=REF_COLOR, lw=0.8, ls="--")
        ax.set_title(ARM_LABELS[arm], loc="left", fontsize=8)
        ax.set_xlabel("per-pair cosine margin")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([pretty(c) for c in order], fontsize=7)
    axes[0].invert_yaxis()
    return fig, None


def fig_h2_shift_vs_separation(res: Results) -> FigResult:
    h2 = res.coupling.get("h2") or {}
    xy = h2.get("per_cell_xy")
    if not xy:
        reason = h2.get("skipped") or "per_cell_xy absent from coupling.json h2"
        return None, f"H2 scatter unavailable: {reason}"
    cells = sorted(xy)
    sep = np.array([xy[c]["y"] for c in cells])  # parent anchor separation
    shift = np.array([xy[c]["x"] for c in cells])  # noise-normalized answer shift
    two_panel = res.dv3_ok
    fig, axes = plt.subplots(1, 2 if two_panel else 1, figsize=(9.0 if two_panel else 5.2, 4.2))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    ax.scatter(sep, shift, s=18, color=MEASURE_COLORS["dv2_tail"], linewidths=0)
    _label_points(ax, sep, shift, [pretty(c) for c in cells])
    ax.set_xlabel("parent anchor separation (ceiling − floor judge contrast)")
    ax.set_ylabel("noise-normalized answer-vector shift")
    obs, ci = h2.get("obs"), h2.get("ci95")
    if obs is not None and ci:
        # Correlation-stat label (the section 3.8 carve-out: the correlation IS the claim).
        ax.text(
            0.02,
            0.98,
            f"Spearman rho = {obs:.2f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
    ax.set_title("Answer-vector shift vs behavioral separation", loc="left", fontsize=9)
    if two_panel:
        ax2 = axes[1]
        goal = [a for a in GOAL_ARMS if a in arms_present(res)]
        for arm in goal:
            pts = [
                (xy[c]["y"], m["mean_margin"])
                for c in cells
                if (m := _per_type_acc(res, arm, c)) is not None
            ]
            if not pts:
                continue
            x2, y2 = zip(*pts)
            ax2.scatter(x2, y2, s=18, color=ARM_COLORS[arm], label=ARM_LABELS[arm], linewidths=0)
        ax2.axhline(0.0, color=REF_COLOR, lw=0.8, ls="--")
        # Shorter than the left panel's label: the full gloss clips at the
        # figure's right edge on the second axes (round-2 caption audit).
        ax2.set_xlabel("parent anchor separation (judge contrast)")
        ax2.set_ylabel("per-type mean cosine margin")
        ax2.set_title("Discrimination margin vs behavioral separation", loc="left", fontsize=9)
        ax2.legend(fontsize=7)
    return fig, None


def _cluster_order(mat: np.ndarray) -> np.ndarray:
    """Presentation-level hierarchical ordering of a cosine matrix."""
    if mat.shape[0] < 3:
        return np.arange(mat.shape[0])
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    sym = (mat + mat.T) / 2.0
    dist = np.clip(1.0 - sym, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return np.asarray(leaves_list(linkage(squareform(dist, checks=False), method="average")))


def fig_cross_type_cosine_heatmaps(res: Results) -> FigResult:
    ct = res.dv1["cross_type"]
    slots = [s for s in ("ce", "pe") if ct.get(s, {}).get("matrix")]
    if not slots:
        return None, "no cross-type matrices in dv1_context_shift.json"
    fig, axes = plt.subplots(
        1, len(slots), figsize=(4.8 * len(slots) + 1.0, 4.8), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    slot_names = {"ce": "context-end", "pe": "prefix-end"}
    im = None
    for ax, slot in zip(axes, slots):
        mat = np.asarray(ct[slot]["matrix"], dtype=float)
        labels = ct[slot]["labels"]
        order = _cluster_order(mat)
        mat = mat[np.ix_(order, order)]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
        if len(labels) <= 30:
            names = [f"{pretty(labels[i][0])} {labels[i][1]}" for i in order]
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=5, rotation=30, ha="right")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=5)
        else:  # 117 rows at production — per-row labels unreadable; ids live in the JSON
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(
            f"mean shift-direction cosine, {slot_names[slot]} "
            f"(hierarchically clustered, n={mat.shape[0]})",
            loc="left",
            fontsize=8,
        )
    assert im is not None
    fig.colorbar(im, ax=list(axes), shrink=0.85, label="cosine")
    return fig, None


def fig_per_layer_accuracy(res: Results) -> FigResult:
    if not res.dv3_ok:
        return None, f"DV3 unavailable: {res.dv3.get('skipped', 'no per_config')}"
    layers = [int(x) for x in res.dv3["meta"]["layers"]]
    per = res.dv3["per_config"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for arm in arms_present(res):
        xs, ys, lo, hi = [], [], [], []
        for layer in layers:
            rec = per.get(f"{arm}|L{layer}|{POOL_PRIMARY}")
            if rec is None:
                continue
            pooled = rec["pooled"][METRIC_PRIMARY]
            xs.append(layer)
            ys.append(pooled["acc"])
            ci = pooled.get("acc_ci95_clustered") or [pooled["acc"], pooled["acc"]]
            lo.append(max(0.0, pooled["acc"] - ci[0]))
            hi.append(max(0.0, ci[1] - pooled["acc"]))
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=np.array([lo, hi]),
            marker="o",
            ms=4,
            lw=1.2,
            capsize=2,
            color=ARM_COLORS[arm],
            label=ARM_LABELS[arm],
        )
    ax.axhline(0.5, color=REF_COLOR, lw=0.8, ls="--")
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("pooled paired two-alternative accuracy (cosine)")
    add_direction_arrow(ax, "y", "up")
    ax.set_title("Pooled discrimination accuracy per layer", loc="left")
    ax.legend(fontsize=7)
    return fig, None


def fig_consistency_vs_band(res: Results) -> FigResult:
    order = type_order_worst_to_best(res)
    panels = [
        ("dv1_ce", res.dv1["per_cell"], "ce"),
        ("dv1_pe", res.dv1["per_cell"], "pe"),
        ("dv2_tail", res.dv2["per_cell"], POOL_PRIMARY),
    ]
    fig, axes = plt.subplots(
        1, 3, figsize=(11.0, _pertype_height(len(order), per=0.22)), sharey=True
    )
    band_labeled = False
    for ax, (name, per_cell, key) in zip(np.atleast_1d(axes), panels):
        for i, cell in enumerate(order):
            rec = per_cell.get(cell, {}).get(key)
            if not isinstance(rec, dict):
                continue
            prim = rec.get("primary", {})
            cons, ci, band = (
                prim.get("consistency"),
                prim.get("consistency_ci95"),
                prim.get("band95"),
            )
            if cons is None:
                continue  # degenerate-at-pe / excluded — no consistency read
            err = _xerr(cons, ci)
            ax.errorbar(
                [cons],
                [i],
                xerr=err,
                fmt="o",
                ms=4,
                lw=1.0,
                capsize=1.5,
                color=MEASURE_COLORS[name],
            )
            if band is not None:
                ax.plot(
                    [band],
                    [i],
                    marker="|",
                    ms=9,
                    markeredgewidth=1.4,
                    color=NULL_COLOR,
                    ls="none",
                    label=("95th pct of label-permutation null" if not band_labeled else None),
                )
                band_labeled = True
        ax.axvline(0.0, color=REF_COLOR, lw=0.6, ls=":")
        ax.set_xlabel("within-type direction consistency")
        ax.set_title(MEASURE_LABELS[name], loc="left", fontsize=8)
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_yticks(range(len(order)))
    ax0.set_yticklabels([pretty(c) for c in order], fontsize=7)
    ax0.invert_yaxis()
    handles, labels = [], []
    for ax in np.atleast_1d(axes):
        h, la = ax.get_legend_handles_labels()
        handles += h
        labels += la
    if handles:
        np.atleast_1d(axes)[-1].legend(handles[:1], labels[:1], fontsize=7, loc="best")
    return fig, None


def fig_knn_retrieval(res: Results) -> FigResult:
    if not res.dv3_ok:
        return None, f"DV3 unavailable: {res.dv3.get('skipped', 'no per_config')}"
    arms = arms_present(res)
    per = res.dv3["per_config"]
    metrics = ("cosine", "euclidean")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    ks = None
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        chance_drawn = False
        for j, arm in enumerate(arms):
            knn = per[f"{arm}|L{res.dv3_layer}|{POOL_PRIMARY}"]["knn"][metric]
            ks = sorted(knn["acc_at_k"], key=int)
            xs = np.arange(len(ks)) + (j - (len(arms) - 1) / 2) * (0.8 / len(arms))
            ax.bar(
                xs,
                [knn["acc_at_k"][k] for k in ks],
                width=0.8 / len(arms) * 0.92,
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm] if metric == "cosine" else None,
            )
            if not chance_drawn:
                for xi, k in enumerate(ks):
                    ax.plot(
                        [xi - 0.42, xi + 0.42],
                        [knn["chance_at_k"][k]] * 2,
                        color="#333333",
                        lw=0.9,
                        ls="--",
                        label="chance (k / pool size)" if xi == 0 and metric == "cosine" else None,
                    )
                chance_drawn = True
        ax.set_xticks(range(len(ks or [])))
        ax.set_xticklabels([f"top-{k}" for k in (ks or [])])
        ax.set_title(f"{metric} retrieval", loc="left", fontsize=9)
        ax.set_xlabel("retrieval tolerance k")
    ax0 = np.atleast_1d(axes)[0]
    ax0.set_ylabel("fraction of contexts whose true answer\nvector is within the k nearest")
    add_direction_arrow(ax0, "y", "up")
    fig.legend(loc="upper right", fontsize=6.5, ncol=1)
    return fig, None


def fig_carrier_transfer(res: Results) -> FigResult:
    if not res.dv3_ok:
        return None, f"DV3 unavailable: {res.dv3.get('skipped', 'no per_config')}"
    per = res.dv3["per_config"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    any_pts = False
    for arm in arms_present(res):
        transfer = per[f"{arm}|L{res.dv3_layer}|{POOL_PRIMARY}"].get("carrier_transfer") or {}
        pts = [
            (rec["own_pair_acc"], rec["cross_carrier_acc"])
            for rec in transfer.values()
            if isinstance(rec, dict) and rec.get("own_pair_acc") is not None
        ]
        if not pts:
            continue
        any_pts = True
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=16, color=ARM_COLORS[arm], label=ARM_LABELS[arm], linewidths=0)
    if not any_pts:
        plt.close(fig)
        return None, "no carrier_transfer records at the registered config"
    ax.plot([0, 1], [0, 1], color=REF_COLOR, lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("own-pair accuracy (per type, cosine)")
    ax.set_ylabel("cross-carrier accuracy (same type + value pair)")
    ax.set_title("Pair-specific vs value-generic discrimination", loc="left")
    ax.legend(fontsize=7)
    return fig, None


def fig_raw_vs_normalized_magnitude(res: Results) -> FigResult:
    panels = [
        (
            "dv1_ce",
            res.dv1["per_cell"],
            "ce",
            res.dv1_primary_idx(),
            "carrier-shift yardstick (median)",
            "median context-vector shift",
        ),
        (
            "dv2_tail",
            res.dv2["per_cell"],
            POOL_PRIMARY,
            res.dv2_primary_idx(),
            "split-half draw-noise floor (median)",
            "median answer-vector shift",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, (name, per_cell, key, idx, xlabel, ylabel) in zip(np.atleast_1d(axes), panels):
        xs, ys, names = [], [], []
        for cell in sorted(per_cell):
            rec = per_cell[cell].get(key)
            if not isinstance(rec, dict) or "yardstick" not in rec:
                continue
            x, y = rec["yardstick"][idx], rec["median_norm"][idx]
            if x and y and x > 0 and y > 0:
                xs.append(x)
                ys.append(y)
                names.append(pretty(cell))
        if xs:
            ax.scatter(xs, ys, s=16, color=MEASURE_COLORS[name], linewidths=0)
            _label_points(ax, xs, ys, names, fontsize=5)
            lims = [min(xs + ys) * 0.8, max(xs + ys) * 1.25]
            ax.plot(lims, lims, color=REF_COLOR, lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(MEASURE_LABELS[name], loc="left", fontsize=9)
    return fig, None


def fig_pooling_twin_deltas(res: Results) -> FigResult:
    two_panel = res.dv3_ok
    fig, axes = plt.subplots(1, 2 if two_panel else 1, figsize=(9.6 if two_panel else 5.0, 4.4))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    xs, ys, names = [], [], []
    for cell in sorted(res.dv2["per_cell"]):
        recs = res.dv2["per_cell"][cell]
        rt = recs.get("tail", {}).get("primary", {}).get("ratio")
        rs = recs.get("span", {}).get("primary", {}).get("ratio")
        if rt and rs and rt > 0 and rs > 0:
            xs.append(rt)
            ys.append(rs)
            names.append(pretty(cell))
    if xs:
        ax.scatter(xs, ys, s=16, color=MEASURE_COLORS["dv2_tail"], linewidths=0)
        _label_points(ax, xs, ys, names, fontsize=5)
        lims = [min(xs + ys) * 0.8, max(xs + ys) * 1.25]
        ax.plot(lims, lims, color=REF_COLOR, lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("noise-normalized shift, tail-inclusive pooling")
    ax.set_ylabel("noise-normalized shift, span-mean pooling")
    ax.set_title("Answer-pooling twins: shift magnitude", loc="left", fontsize=9)
    if two_panel:
        ax2 = axes[1]
        per = res.dv3["per_config"]
        for arm in [a for a in ("779ce", "1738pe", "1738ce") if a in arms_present(res)]:
            tail_pt = per[f"{arm}|L{res.dv3_layer}|tail"]["per_type"]
            span_pt = per.get(f"{arm}|L{res.dv3_layer}|span", {}).get("per_type", {})
            pts = []
            for cell, rec in tail_pt.items():
                mt = rec.get(METRIC_PRIMARY) if isinstance(rec, dict) else None
                s_rec = span_pt.get(cell)
                ms = s_rec.get(METRIC_PRIMARY) if isinstance(s_rec, dict) else None
                if isinstance(mt, dict) and isinstance(ms, dict):
                    pts.append((mt["acc"], ms["acc"]))
            if not pts:
                continue
            x2, y2 = zip(*pts)
            ax2.scatter(x2, y2, s=16, color=ARM_COLORS[arm], label=ARM_LABELS[arm], linewidths=0)
        ax2.plot([0, 1], [0, 1], color=REF_COLOR, lw=0.8, ls="--")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("per-type accuracy, tail-inclusive targets")
        ax2.set_ylabel("per-type accuracy, span-mean targets")
        ax2.set_title("Answer-pooling twins: discrimination", loc="left", fontsize=9)
        ax2.legend(fontsize=7)
    return fig, None


# ── registry + entrypoint ──────────────────────────────────────────────

FIGURES: list[tuple[str, Callable[[Results], FigResult]]] = [
    ("hero1_per_type_2afc", fig_hero1_per_type_2afc),
    ("hero2_shift_ratio_per_type", fig_hero2_shift_ratio),
    ("expl_margin_scatter_per_type", fig_margin_scatter_per_type),
    ("expl_h2_shift_vs_separation", fig_h2_shift_vs_separation),
    ("expl_cross_type_cosine_heatmaps", fig_cross_type_cosine_heatmaps),
    ("expl_per_layer_accuracy", fig_per_layer_accuracy),
    ("expl_consistency_vs_band", fig_consistency_vs_band),
    ("expl_knn_retrieval", fig_knn_retrieval),
    ("expl_carrier_transfer", fig_carrier_transfer),
    ("expl_raw_vs_normalized_magnitude", fig_raw_vs_normalized_magnitude),
    ("expl_pooling_twin_deltas", fig_pooling_twin_deltas),
]


def render_all(results_dir: Path | str, out_dir: Path | str) -> dict[str, dict]:
    """Render every registry figure from the Phase C outputs; returns a
    manifest {stem: {written, path?, skipped?}}. Skips are RECORDED, never
    silent; zero rendered figures is a hard failure."""
    res = load_results(Path(results_dir))
    set_paper_style("blog")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for stem, fn in FIGURES:
        fig, skip = fn(res)
        if fig is None:
            assert skip, f"{stem}: figure fn returned neither a figure nor a skip reason"
            logger.info("[figures] %s SKIPPED: %s", stem, skip)
            manifest[stem] = {"written": False, "skipped": skip}
            continue
        paths = savefig_paper(fig, stem, dir=str(out))
        plt.close(fig)
        manifest[stem] = {"written": True, "path": str(paths["png"]), "skipped": None}
        logger.info("[figures] %s -> %s", stem, paths["png"])
    assert any(v["written"] for v in manifest.values()), (
        f"no figure rendered from {results_dir} — Phase C outputs malformed?"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Issue #2215 figures from the Phase C outputs (no recomputation)."
    )
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--results-dir", type=Path, default=repo / "eval_results" / "issue_2215")
    ap.add_argument("--out-dir", type=Path, default=repo / "figures" / "issue_2215")
    args = ap.parse_args(argv)
    manifest = render_all(args.results_dir, args.out_dir)
    n = sum(1 for v in manifest.values() if v["written"])
    print(f"[figures] {n}/{len(manifest)} figures written to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
