#!/usr/bin/env python3
"""Issue #2162 — manifest-driven figure set (plan §6 report figures).

Consumes ONLY the analysis outputs under ``eval_results/issue_2162/f_metrics/``
and renders to ``figures/issue_2162/``:

Manifest coverage (planned_manifest.json figure id -> producer key):

- ``per_type_f_beh`` + ``per_type_f_beh_perpair``      -> ``hero``
- ``read_write_2x2``                                   -> ``two_by_two``
- ``probe_layer_curves``                               -> ``layer_profile``
  (per type x slot AUC-vs-layer curves with the within-carrier permutation
  band + per-value-pair points; the heatmap ships as a companion view)
- ``layer_profile_stage2`` + ``_perpair``              -> ``stage2_layer_profile``
- ``route_contrasts``                                  -> ``route_contrasts``
- ``route_contrasts_perpair``                          -> ``route_contrasts_perpair``
- ``recency_load_curves``                              -> ``dose_position``
  (shuffled-null band behind + the registered per-pair slope CI annotated)
- ``recency_load_perpair``                             -> ``recency_load_perpair``
- ``anchor_separation_diag``                           -> ``anchor_separation``
- ``act_beh_agreement``                                -> ``act_beh_agreement``
- ``margin_validation``                                -> ``margin_validation``
- ``crosstype_null_by_donor``                          -> ``crosstype_by_donor``
- ``coherence_caphit``                                 -> ``diagnostics``
  (per-arm EXCESS incoherence over the anchor baseline + cap-hit w/ 2% line)

Note on ``route_contrasts`` conflict cells: the manifest's "balance shift =
(judge_demo - judge_instr)/100 normalized floor-to-ceiling" IS the f_beh
reduction for conflict pairs (value_b = the demo-carried value, value_a = the
instruction-carried value), so the plotted quantity matches the transform.

Every PNG gets a ``.meta.json`` sidecar (inputs + git provenance). Errorbar
offsets are non-negative by construction (the xerr/yerr gotcha). Later-phase
inputs not yet on disk (stage-2, margin) drop their keys from the default
``--only all`` set with a loud log line; explicitly requesting them without
the input is a hard failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2162.figures")

FAMILY_COLORS = {"P1": "#4878d0", "P2": "#ee854a", "P3": "#6acc64", None: "#9d9d9d"}
ARM_COLORS = {"steered": "#4878d0", "shuffled": "#9d9d9d", "crosstype": "#c44e52"}


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _save(fig, out_dir: Path, name: str, inputs: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "figure": name,
        "inputs": [str(p) for p in inputs],
        **as_metadata_dict(git_provenance()),
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("[figures] wrote %s", png)


def _err(lo: float | None, hi: float | None, v: float) -> tuple[float, float]:
    """Non-negative errorbar offsets from CI bounds (NaN when absent)."""
    if lo is None or hi is None:
        return float("nan"), float("nan")
    return max(0.0, v - lo), max(0.0, hi - v)


def _cells_sorted(per_cell: dict) -> list[tuple[str, dict]]:
    fam_rank = {"P1": 0, "P2": 1, "P3": 2, None: 3}
    return sorted(
        per_cell.items(),
        key=lambda kv: (fam_rank.get(kv[1]["family"], 3), kv[1]["cell"], kv[1]["slot"]),
    )


# Manifest separation exclusion bar — pinned equal to
# issue2162_analysis.SEPARATION_BAR by test_issue2162_figures.py (a local
# constant so the figure script does not import the torch/scipy-heavy
# analysis module at render time).
SEPARATION_BAR = 0.5


def _perpair_surviving(
    arm_rows: dict[str, list[dict]],
) -> dict[tuple[str, str, str], list[tuple[str, float]]]:
    """Manifest ``per_type_f_beh_perpair`` selection (r2 R1): SAME exclusion
    as ``per_type_f_beh`` (|separation| >= SEPARATION_BAR), NO aggregation —
    one (pair_id, f_beh) point per SURVIVING pair, keyed (cell, slot, arm)."""
    out: dict[tuple[str, str, str], list[tuple[str, float]]] = defaultdict(list)
    for arm, rows in arm_rows.items():
        for r in rows:
            if (
                r.get("f_beh") is not None
                and r.get("separation") is not None
                and abs(r["separation"]) >= SEPARATION_BAR
            ):
                out[(r["cell"], r["slot"], arm)].append((r["pair_id"], r["f_beh"]))
    return out


def fig_hero(
    stats: dict, arm_rows: dict[str, list[dict]], out_dir: Path, inputs: list[Path]
) -> None:
    """Manifest ``per_type_f_beh`` (one PANEL PER SLOT, post-exclusion n per
    type — r2 R3) + ``per_type_f_beh_perpair`` (same exclusion, one point per
    surviving pair PER ARM, pair-id labeled — r2 R1)."""
    items = _cells_sorted(stats["per_cell"])
    slots = [s for s in ("ce", "pe") if any(r["slot"] == s for _, r in items)]
    width = 0.27
    n_widest = max((sum(1 for _, r in items if r["slot"] == s) for s in slots), default=1)
    fig, axes = plt.subplots(
        len(slots),
        1,
        figsize=(max(14, n_widest * 0.5), 5.5 * len(slots)),
        sharey=True,
        squeeze=False,
    )
    for ax, slot in zip(axes.ravel(), slots, strict=True):
        s_items = [(key, r) for key, r in items if r["slot"] == slot]
        labels = [f"{r['cell']}\n(n={r['n_post_exclusion']})" for _, r in s_items]
        x = np.arange(len(s_items))
        for k, arm in enumerate(("steered", "shuffled", "crosstype")):
            vals, lo_off, hi_off = [], [], []
            for _, r in s_items:
                v = r.get(f"f_{arm}_mean")
                ci = (r.get("ci95") or {}).get(arm, [None, None])
                vals.append(np.nan if v is None else v)
                e = _err(ci[0], ci[1], v if v is not None else 0.0)
                lo_off.append(e[0])
                hi_off.append(e[1])
            ax.bar(
                x + (k - 1) * width,
                vals,
                width,
                yerr=[lo_off, hi_off],
                color=ARM_COLORS[arm],
                label=arm,
                error_kw={"lw": 0.7},
            )
        for i, (_, r) in enumerate(s_items):
            if r["untestable_causal"]:
                ax.text(x[i], 0.02, "n/a", ha="center", fontsize=5, rotation=90, color="#555")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.set_ylabel("F_beh (mean over post-exclusion pairs)")
        ax.set_title(f"slot = {slot}")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Per-type-cell F_beh by slot: steered vs both nulls "
        "(95% pair-clustered CIs; n = post-exclusion pairs)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # rotated labels must clear the next panel
    _save(fig, out_dir, "hero_ftype", inputs)

    # Per-unit companion (r2 R1): SAME separation exclusion, one point per
    # surviving pair PER ARM (arm-offset like the bars), pair-id labeled.
    perpair = _perpair_surviving(arm_rows)
    rng = np.random.default_rng(2162)
    fig, axes = plt.subplots(
        len(slots),
        1,
        figsize=(max(14, n_widest * 0.5), 5.5 * len(slots)),
        sharey=True,
        squeeze=False,
    )
    for ax, slot in zip(axes.ravel(), slots, strict=True):
        s_items = [(key, r) for key, r in items if r["slot"] == slot]
        labels = [r["cell"] for _, r in s_items]
        x = np.arange(len(s_items))
        for i, (_, rcell) in enumerate(s_items):
            for k, arm in enumerate(("steered", "shuffled", "crosstype")):
                for pair_id, f in perpair.get((rcell["cell"], slot, arm), []):
                    xi = i + (k - 1) * width + float(rng.uniform(-0.06, 0.06))
                    ax.scatter(xi, f, s=6, alpha=0.6, color=ARM_COLORS[arm])
                    ax.annotate(pair_id, (xi, f), fontsize=3.2, alpha=0.7)
        for k, arm in enumerate(("steered", "shuffled", "crosstype")):
            ax.scatter([], [], s=10, color=ARM_COLORS[arm], label=arm)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.set_ylabel("per-pair F_beh")
        ax.set_title(f"slot = {slot}")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Per-pair F_beh, separation-excluded survivors only, per arm "
        "(no aggregation; pair-id labeled)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # rotated labels must clear the next panel
    _save(fig, out_dir, "hero_ftype_perpair", inputs)


# Manifest ``read_write_2x2`` quadrant labels (r3 MAJOR 1). Quadrant
# membership is the PERSISTED (probe_verdict, causal_verdict) pair from
# two_by_two.json — never plot position: the probe-positive threshold is the
# PER-CELL max-selected 97.5th permutation band (probe.json perm_band_97p5),
# so no single vertical line marks the read threshold and none is drawn (an
# AUC=0.5 chance line would misread as the registered threshold).
QUADRANT_LABELS = {
    ("positive", "positive"): "stored-and-used",
    ("positive", "null"): "stored-but-unusable",
    ("null", "positive"): "used-but-not-decoded",
    ("null", "null"): "absent",
}
QUADRANT_STYLE = {
    "stored-and-used": ("o", "#2a9d2a"),
    "stored-but-unusable": ("s", "#4878d0"),
    "used-but-not-decoded": ("D", "#ee854a"),
    "absent": ("o", "#9d9d9d"),
    "untestable-causal": ("x", "#c44e52"),
}


def _quadrant_of(r: dict) -> str | None:
    """Registered quadrant label for a persisted two_by_two.json row.

    ``untestable-causal`` (post-exclusion n < 12) is the explicit fifth
    label regardless of the probe verdict; a ``missing`` probe verdict (no
    probe rows for the cell) maps to None — such rows carry no ``max_auc``
    and cannot be positioned in the AUC-vs-F plane.
    """
    if r["causal_verdict"] == "untestable-causal":
        return "untestable-causal"
    return QUADRANT_LABELS.get((r["probe_verdict"], r["causal_verdict"]))


def fig_two_by_two(two: dict, out_dir: Path, inputs: list[Path]) -> None:
    """Manifest ``read_write_2x2``: one point per (type x slot) in the
    AUC-vs-F plane, quadrant membership encoded from the persisted
    (probe_verdict, causal_verdict) pair — probe-positive = the per-cell
    max-selected permutation band, causal-positive = Holm +
    disjoint-both-nulls — with the four registered quadrant labels drawn
    (legend, every class always present) and untestable-causal as the
    explicit fifth label. No vertical threshold line: the probe threshold
    is per-cell, so a single vertical (e.g. chance 0.5) would misrepresent
    the registered read threshold (r3 MAJOR 1). Rows with NO steered F
    (zero post-exclusion steered pairs — always untestable-causal) are
    omitted from the scatter and counted in the title, never plotted at a
    fabricated 0.0 (r4 MAJOR 1)."""
    rows = two["cells"]
    plottable = [r for r in rows if r["max_auc"] is not None]
    n_no_probe = len(rows) - len(plottable)
    # r4 MAJOR 1 (round 5): f_steered_mean is None exactly when ZERO steered
    # pairs survived the separation exclusion — the registered y-value does
    # not exist for the row, so the point is OMITTED and counted in the title
    # (mirroring fig_hero's explicit "n/a" and fig_route_contrasts'
    # skip-on-None). A 0.0 substitute would sit on the zero-effect line under
    # the same untestable-causal marker as genuinely-measured small-n means,
    # indistinguishable by position or marker.
    plotted = [r for r in plottable if r["f_steered_mean"] is not None]
    n_no_f = len(plottable) - len(plotted)
    fig, ax = plt.subplots(figsize=(9, 8))
    for quad, (marker, color) in QUADRANT_STYLE.items():
        pts = [r for r in plotted if _quadrant_of(r) == quad]
        ax.scatter(
            [r["max_auc"] for r in pts],
            [r["f_steered_mean"] for r in pts],
            marker=marker,
            s=30,
            color=color,
            label=f"{quad} (n={len(pts)})",
            alpha=0.85,
        )
        for r in pts:
            ax.annotate(
                f"{r['cell']}|{r['slot']}",
                (r["max_auc"], r["f_steered_mean"]),
                fontsize=4.5,
                alpha=0.8,
            )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("probe max-over-layers AUC (read; positive = clears the per-cell perm band)")
    ax.set_ylabel("steered F_beh mean (write)")
    title = "Read x write 2x2 — quadrant = persisted (probe, causal) verdicts"
    if n_no_probe:
        title += f" ({n_no_probe} cells without probe rows omitted)"
    if n_no_f:
        title += f" ({n_no_f} untestable cells without steered F omitted)"
    ax.set_title(title)
    ax.legend(fontsize=8, title="quadrant (verdict-encoded, per-cell probe threshold)")
    _save(fig, out_dir, "two_by_two", inputs)


def fig_layer_profile(probe: dict, perm_npz: Path, out_dir: Path, inputs: list[Path]) -> None:
    """Manifest ``probe_layer_curves``: per (type x slot) AUC-vs-layer curves
    with the within-carrier label-permutation 95% band (per layer, no max
    taken in this view) + per-value-pair thin curves/points; the heatmap
    ships as a compact companion view."""
    results = probe["results"]
    perm = np.load(perm_npz) if perm_npz.exists() else None
    n_layers = len(results[0]["auc_per_layer"])
    xs = np.arange(n_layers)
    for slot in ("ce", "pe"):
        rows = sorted((r for r in results if r["slot"] == slot), key=lambda r: r["cell"])
        if not rows:
            continue
        ncol = 7
        nrow = (len(rows) + ncol - 1) // ncol
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(ncol * 2.4, nrow * 1.9), sharex=True, sharey=True
        )
        axf = np.atleast_1d(axes).ravel()
        for ax, r in zip(axf, rows):
            key = f"{r['cell']}|{slot}"
            if perm is not None and key in perm.files:
                lo, hi = np.percentile(perm[key], [2.5, 97.5], axis=0)
                ax.fill_between(xs, lo, hi, color="#cccccc", alpha=0.6, lw=0)
            for vp_curve in r.get("auc_per_layer_per_vp", []):
                ax.plot(xs, vp_curve, color="#9ecae1", lw=0.6, alpha=0.85)
                ax.scatter(xs, vp_curve, s=2, color="#9ecae1", alpha=0.6)
            ax.plot(xs, r["auc_per_layer"], color="#4878d0", lw=1.3)
            ax.axhline(0.5, color="k", lw=0.5, ls=":")
            ax.set_title(r["cell"], fontsize=6)
            ax.set_ylim(0.0, 1.05)
        for ax in axf[len(rows) :]:
            ax.axis("off")
        fig.suptitle(
            f"probe AUC vs layer — slot {slot} (blue = macro over value-pairs; "
            "thin light curves/points = per value-pair; grey = within-carrier "
            "permutation 95% band per layer)",
            fontsize=9,
        )
        _save(fig, out_dir, f"probe_layer_curves_{slot}", inputs)
    # Companion heatmap (the compact per-layer summary).
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, slot in zip(axes, ("ce", "pe"), strict=True):
        rows = [r for r in results if r["slot"] == slot]
        rows.sort(key=lambda r: r["cell"])
        mat = np.array([r["auc_per_layer"] for r in rows])
        im = ax.imshow(mat, aspect="auto", vmin=0.3, vmax=1.0, cmap="viridis")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["cell"] for r in rows], fontsize=5.5)
        ax.set_xlabel("layer")
        ax.set_title(f"probe AUC per layer — slot {slot}")
    fig.colorbar(im, ax=axes, shrink=0.7, label="LOCO AUC")
    _save(fig, out_dir, "layer_profile", inputs)


def fig_route_contrasts(stats: dict, out_dir: Path, inputs: list[Path]) -> None:
    from explore_persona_space.experiments.issue2162 import bank2162 as B

    per_cell = stats["per_cell"]
    route_pairs = [
        ("instr_format", "demo_format"),
        ("persona_prompted", "demo_persona"),
        ("instr_language", "language_implied"),
        ("persona_prompted", "persona_role_header"),
    ]
    conflict = [c for c in B.all_cells() if c.startswith("conflict_")]
    groups = [(a, b) for a, b in route_pairs] + [(B.base_type_of(c), c) for c in conflict]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = 0.0
    ticks, tick_labels = [], []
    for base, variant in groups:
        for slot in ("ce", "pe"):
            for off, cell, color in ((0.0, base, "#4878d0"), (0.35, variant, "#ee854a")):
                r = per_cell.get(f"{cell}|{slot}")
                if r is None or r.get("f_steered_mean") is None:
                    continue
                ci = (r.get("ci95") or {}).get("steered", [None, None])
                lo, hi = _err(ci[0], ci[1], r["f_steered_mean"])
                ax.bar(
                    x + off,
                    r["f_steered_mean"],
                    0.32,
                    yerr=[[lo], [hi]],
                    color=color,
                    error_kw={"lw": 0.7},
                )
            ticks.append(x + 0.17)
            tick_labels.append(f"{variant}|{slot}")
            x += 1.0
        x += 0.5
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_ylabel("steered F_beh")
    ax.set_title("P2 route contrasts: base type (blue) vs route variant / conflict (orange)")
    _save(fig, out_dir, "route_contrasts", inputs)


def fig_dose_position(stats: dict, out_dir: Path, inputs: list[Path]) -> None:
    """Manifest ``recency_load_curves``: steered F_beh vs depth/load with the
    shuffled-donor null as a shaded BAND behind each curve (95% pair-clustered
    CI, mean line inside — r2 R4) + the registered per-pair slope with its
    pair-clustered bootstrap 95% CI annotated."""
    from explore_persona_space.experiments.issue2162 import bank2162 as B

    per_cell = stats["per_cell"]
    dose_slopes = stats.get("dose_slopes") or {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, prefix, xlab in zip(
        axes, ("recency", "load"), ("history depth d", "distractor load l"), strict=True
    ):
        bases = sorted({B.base_type_of(c) for c in B.crossed_cells() if c.startswith(prefix)})
        for base in bases:
            for slot, ls in (("ce", "-"), ("pe", "--")):
                xs, ys, lo_off, hi_off = [], [], [], []
                null_ys, null_lo, null_hi = [], [], []
                depth_tag = "d" if prefix == "recency" else "l"
                for depth in (1, 3, 5):
                    key = (
                        f"{base}|{slot}"
                        if depth == 1
                        else f"{prefix}_{base}_{depth_tag}{depth}|{slot}"
                    )
                    r = per_cell.get(key)
                    if r is None or r.get("f_steered_mean") is None:
                        continue
                    xs.append(depth)
                    ys.append(r["f_steered_mean"])
                    null_ys.append(r.get("f_shuffled_mean"))
                    nci = (r.get("ci95") or {}).get("shuffled", [None, None])
                    null_lo.append(nci[0])
                    null_hi.append(nci[1])
                    ci = (r.get("ci95") or {}).get("steered", [None, None])
                    e = _err(ci[0], ci[1], ys[-1])
                    lo_off.append(e[0])
                    hi_off.append(e[1])
                if xs:
                    # Shuffled-donor null behind: shaded 95%-CI band + mean
                    # line (r2 R4 — a bare mean line is not a band).
                    band_ok = [
                        m is not None and lo is not None and hi is not None
                        for m, lo, hi in zip(null_ys, null_lo, null_hi, strict=True)
                    ]
                    if any(band_ok):
                        bx = [v for v, ok in zip(xs, band_ok, strict=True) if ok]
                        blo = [v for v, ok in zip(null_lo, band_ok, strict=True) if ok]
                        bhi = [v for v, ok in zip(null_hi, band_ok, strict=True) if ok]
                        bm = [v for v, ok in zip(null_ys, band_ok, strict=True) if ok]
                        ax.fill_between(bx, blo, bhi, color="#bbbbbb", alpha=0.35, lw=0, zorder=1)
                        ax.plot(bx, bm, color="#bbbbbb", ls=ls, lw=0.8, zorder=1)
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=[lo_off, hi_off],
                        marker="o",
                        ls=ls,
                        ms=3.5,
                        label=f"{base}|{slot}",
                        lw=1.1,
                        capsize=2,
                        zorder=2,
                    )
        slope_lines = []
        for k in sorted(dose_slopes):
            pfx, base, slot = k.split("|")
            if pfx != prefix:
                continue
            v = dose_slopes[k]
            slope_lines.append(
                f"{base}|{slot}: {v['slope_mean']:+.3f} "
                f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] (n={v['n_pairs']})"
            )
        if slope_lines:
            ax.text(
                0.02,
                0.02,
                "per-pair slope, pair-clustered 95% CI:\n" + "\n".join(slope_lines),
                transform=ax.transAxes,
                fontsize=5,
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.7, "lw": 0.3},
            )
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel(xlab + " (1 = uncrossed base cell)")
        ax.set_title(f"{prefix} curves (steered F_beh; grey band = shuffled-donor null 95% CI)")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("steered F_beh")
    _save(fig, out_dir, "dose_position", inputs)


def fig_margin_validation(
    margin_cells: list[dict],
    f_cells: list[dict],
    validation: dict,
    out_dir: Path,
    inputs: list[Path],
) -> None:
    """Manifest ``margin_validation``: the REGISTERED per-(cell x slot) mean
    scatter (dynamic-range-screened points from margin_validation.json — the
    grain the rule-19 rho is computed on) with the per-pair points behind as
    the low-level companion (r2 R2)."""
    f_by_key = {(r["pair_id"], r["slot"]): r["f_beh"] for r in f_cells if r["f_beh"] is not None}
    xs, ys = [], []
    for r in margin_cells:
        if r["arm"] != "steered" or r.get("margin_shift") is None:
            continue
        f = f_by_key.get((r["pair_id"], r["slot"]))
        if f is not None:
            xs.append(r["margin_shift"])
            ys.append(f)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xs, ys, s=8, alpha=0.25, color="#9d9d9d", label="per-pair (companion)")
    pc = validation.get("percell_points") or []
    ax.scatter(
        [p["margin_shift_mean"] for p in pc],
        [p["f_beh_mean"] for p in pc],
        s=26,
        alpha=0.9,
        color="#4878d0",
        label="per-(cell x slot) mean (registered grain)",
    )
    for p in pc:
        ax.annotate(
            f"{p['cell']}|{p['slot']}",
            (p["margin_shift_mean"], p["f_beh_mean"]),
            fontsize=3.5,
            alpha=0.8,
        )
    rho_c = validation.get("rho_margin_fbeh_percell")
    rho_p = validation.get("rho_margin_fbeh_perpair")
    ax.set_xlabel("TF fixed-pool margin shift (patched - floor anchor)")
    ax.set_ylabel("F_beh (steered)")
    ax.legend(fontsize=7)
    ax.set_title(
        f"Margin validation (rule 19): per-cell rho={rho_c if rho_c is None else round(rho_c, 3)} "
        f"(n_cells={validation.get('n_cells')}, validated={validation.get('validated')}); "
        f"per-pair rho={rho_p if rho_p is None else round(rho_p, 3)} "
        f"(n_pairs={validation.get('n_pairs')})"
    )
    _save(fig, out_dir, "margin_validation", inputs)


def fig_diagnostics(
    stats: dict,
    arm_rows: dict[str, list[dict]],
    anchor_rows: list[dict],
    out_dir: Path,
    inputs: list[Path],
) -> None:
    """Manifest ``coherence_caphit``: per (cell x slot x arm) EXCESS
    incoherence — incoherent fraction (score <= 60) MINUS the cell's anchor
    baseline incoherent rate — plus cap-hit fraction with the 2% re-gen
    trigger line, and the post-exclusion-n survival panel."""
    # Anchor baseline incoherent rate per cell, deduped per (carrier, value)
    # context (adjacent value-pairs share contexts; row fields repeat them).
    ctx_counts: dict[str, dict[tuple[str, str], tuple[int, int]]] = defaultdict(dict)
    for a in anchor_rows:
        for side in ("floor", "ceiling"):
            val = a["value_a"] if side == "floor" else a["value_b"]
            tot = a.get(f"n_{side}_rollouts")
            coh = a.get(f"n_{side}_coherent")
            # r4 MINOR 1 (round 5): an ABSENT coherent count is not "0 coherent
            # of N" (that fabricates a maximally-incoherent baseline) — skip
            # the context; a cell left with no valid contexts takes the r2 H3
            # missing-baseline NaN + loud-warning path below.
            if tot and coh is not None:
                ctx_counts[a["cell"]][(a["carrier"], val)] = (int(tot), int(coh))
    anchor_incoh: dict[str, float] = {}
    for cell, ctxs in ctx_counts.items():
        tot = sum(t for t, _ in ctxs.values())
        coh = sum(c for _, c in ctxs.values())
        if tot:
            anchor_incoh[cell] = 1.0 - coh / tot

    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"coh": 0, "cap": 0, "n": 0})
    for arm, rows in arm_rows.items():
        for r in rows:
            k = (f"{r['cell']}|{r['slot']}", arm)
            agg[k]["coh"] += r["n_coherent"]
            agg[k]["cap"] += r["n_cap_hit"]
            agg[k]["n"] += r["n_draws"]
    items = _cells_sorted(stats["per_cell"])
    labels = [f"{r['cell']}|{r['slot']}" for _, r in items]
    x = np.arange(len(labels))
    width = 0.27
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(labels) * 0.32), 10), sharex=True)
    missing_baseline: set[str] = set()
    for k, arm in enumerate(("steered", "shuffled", "crosstype")):
        excess, cap = [], []
        for (key, _), rec in [((lab, arm), agg.get((lab, arm))) for lab in labels]:
            if rec is None or rec["n"] == 0:
                excess.append(np.nan)
                cap.append(np.nan)
                continue
            cell = key.split("|")[0]
            incoh = 1.0 - rec["coh"] / rec["n"]
            base = anchor_incoh.get(cell)
            if base is None:
                # r2 H3: a MISSING anchor baseline (legacy anchors.jsonl rows
                # without n_*_rollouts) is NaN + a loud log — never a silent
                # 0.0 substitute that fakes "no excess over baseline".
                if cell not in missing_baseline:
                    missing_baseline.add(cell)
                    logger.warning(
                        "[figures] no anchor incoherence baseline for cell %s — excess "
                        "plotted as NaN (legacy anchors.jsonl without n_*_rollouts?)",
                        cell,
                    )
                excess.append(np.nan)
            else:
                excess.append(incoh - base)
            cap.append(rec["cap"] / rec["n"])
        axes[0].bar(x + (k - 1) * width, excess, width, color=ARM_COLORS[arm], label=arm)
        axes[1].bar(x + (k - 1) * width, cap, width, color=ARM_COLORS[arm], label=arm)
    axes[0].axhline(0.0, color="k", lw=0.6)
    axes[0].set_ylabel("excess incoherence\n(arm - anchor baseline)")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0.02, color="k", lw=0.6, ls=":")
    axes[1].set_ylabel("cap-hit fraction")
    n_post = [r["n_post_exclusion"] for _, r in items]
    axes[2].bar(x, n_post, color="#6acc64")
    axes[2].axhline(12, color="k", lw=0.6, ls=":")
    axes[2].set_ylabel("post-exclusion n (pairs)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=90, fontsize=5.5)
    fig.suptitle(
        "Diagnostics: excess incoherence over anchor baseline (score<=60), "
        "cap-hit (2% re-gen line), separation survival (floor 12)"
    )
    _save(fig, out_dir, "diagnostics", inputs)


def fig_route_contrasts_perpair(f_cells: list[dict], out_dir: Path, inputs: list[Path]) -> None:
    """Manifest ``route_contrasts_perpair``: no aggregation — per-pair steered
    F_beh points for every route-variant/conflict cell beside its base type,
    labeled by pair id."""
    from explore_persona_space.experiments.issue2162 import bank2162 as B

    route_pairs = [
        ("instr_format", "demo_format"),
        ("persona_prompted", "demo_persona"),
        ("instr_language", "language_implied"),
        ("persona_prompted", "persona_role_header"),
    ]
    conflict = [c for c in B.all_cells() if c.startswith("conflict_")]
    groups = list(route_pairs) + [(B.base_type_of(c), c) for c in conflict]
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in f_cells:
        if r["f_beh"] is not None:
            by_cell[(r["cell"], r["slot"])].append(r)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = 0.0
    ticks, tick_labels = [], []
    rng = np.random.default_rng(2162)
    for base, variant in groups:
        for slot in ("ce", "pe"):
            for off, cell, color in ((0.0, base, "#4878d0"), (0.35, variant, "#ee854a")):
                rows = by_cell.get((cell, slot), [])
                for r in rows:
                    xi = x + off + float(rng.uniform(-0.06, 0.06))
                    ax.scatter(xi, r["f_beh"], s=7, color=color, alpha=0.65)
                    ax.annotate(r["pair_id"], (xi, r["f_beh"]), fontsize=3.2, alpha=0.7)
            ticks.append(x + 0.17)
            tick_labels.append(f"{variant}|{slot}")
            x += 1.0
        x += 0.5
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_ylabel("per-pair F_beh (steered)")
    ax.set_title(
        "Route contrasts, per-pair points: base type (blue) vs route variant / "
        "conflict (orange); pair-id labeled"
    )
    _save(fig, out_dir, "route_contrasts_perpair", inputs)


def fig_recency_load_perpair(f_cells: list[dict], out_dir: Path, inputs: list[Path]) -> None:
    """Manifest ``recency_load_perpair``: per-pair steered F_beh trajectories
    across depth/load levels — lines connect the SAME (carrier x value-pair)
    across levels (level 1 = the uncrossed base cell)."""
    from explore_persona_space.experiments.issue2162 import bank2162 as B

    crossed = set(B.crossed_cells())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, prefix, tag in zip(axes, ("recency", "load"), ("d", "l"), strict=True):
        traj: dict[tuple[str, str, str, str, str], dict[int, float]] = defaultdict(dict)
        for r in f_cells:
            if r["f_beh"] is None:
                continue
            base = B.base_type_of(r["cell"])
            if r["cell"] == base:
                if not any(c.startswith(f"{prefix}_{base}_") for c in crossed):
                    continue
                level = 1
            elif r["cell"].startswith(f"{prefix}_{base}_{tag}"):
                level = int(r["cell"].rsplit(tag, 1)[-1])
            else:
                continue
            traj[(base, r["slot"], r["carrier"], r["value_a"], r["value_b"])][level] = r["f_beh"]
        bases = sorted({k[0] for k in traj})
        cmap = plt.get_cmap("tab10")
        base_color = {b: cmap(i % 10) for i, b in enumerate(bases)}
        for (base, slot, _, _, _), by_level in sorted(traj.items()):
            xs = sorted(by_level)
            ys = [by_level[lv] for lv in xs]
            ls = "-" if slot == "ce" else "--"
            ax.plot(xs, ys, color=base_color[base], ls=ls, lw=0.6, alpha=0.45)
            ax.scatter(xs, ys, s=6, color=base_color[base], alpha=0.6)
        for b in bases:
            ax.plot([], [], color=base_color[b], label=b)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel(f"{prefix} level (1 = base cell; ce solid, pe dashed)")
        ax.set_title(f"{prefix}: per-pair F_beh trajectories")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("per-pair F_beh (steered)")
    _save(fig, out_dir, "recency_load_perpair", inputs)


def fig_anchor_separation(anchor_rows: list[dict], out_dir: Path, inputs: list[Path]) -> None:
    """Manifest ``anchor_separation_diag``: per-pair anchor separation strip
    per type with the +/-0.5 exclusion bars and pre/post-exclusion counts."""
    cells = sorted({a["cell"] for a in anchor_rows})
    fig, ax = plt.subplots(figsize=(max(12, len(cells) * 0.32), 6))
    rng = np.random.default_rng(2162)
    for i, cell in enumerate(cells):
        seps = [a["separation"] for a in anchor_rows if a["cell"] == cell]
        vals = [s for s in seps if s is not None]
        xs = i + rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(xs, vals, s=7, color="#4878d0", alpha=0.6)
        n_kept = sum(1 for v in vals if abs(v) >= 0.5)
        ax.text(
            i,
            1.02,
            f"{n_kept}/{len(seps)}",
            ha="center",
            fontsize=5,
            transform=ax.get_xaxis_transform(),
        )
    for y in (0.5, -0.5):
        ax.axhline(y, color="#c44e52", lw=0.8, ls=":")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells, rotation=90, fontsize=6)
    ax.set_ylabel("anchor separation (ceiling - floor, judge-contrast units)")
    ax.set_title(
        "Anchor separation per pair (K=10 draws; |sep| >= 0.5 keeps the pair; "
        "kept/total per type above)"
    )
    _save(fig, out_dir, "anchor_separation_diag", inputs)


# r3 MINOR 1: the manifest transform registers "Spearman rho across cells
# WITH DYNAMIC RANGE reported in-panel" — the same restriction phrase as the
# rule-19 grain, so the screen mirrors
# issue2162_analysis.RULE19_DYNAMIC_RANGE_SCREEN.
ACT_BEH_DYNAMIC_RANGE_SCREEN = (
    "a (cell x slot) unit enters an arm's rho iff it has >=2 separation-kept rows with both "
    "F_act and F_beh present AND nonzero spread (max > min) in BOTH quantities across those "
    "rows (a constant/degenerate unit carries no dynamic range)"
)


def _act_beh_units(rows: list[dict]) -> dict[str, dict]:
    """Per-(cell|slot) aggregation for ``act_beh_agreement`` (r3 MINOR 1).

    Applies the manifest separation exclusion, then the rule-19-mirrored
    dynamic-range screen; returns per-unit means + ``in_rho`` (whether the
    unit enters the arm's Spearman rho). Screened-out units stay PLOTTED —
    the manifest plotted_quantity is one point per cell-arm — they are only
    excluded from the statistic.
    """
    per: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if (
            r["f_beh"] is None
            or r["f_act"] is None
            or r["separation"] is None
            or abs(r["separation"]) < SEPARATION_BAR
        ):
            continue
        per[f"{r['cell']}|{r['slot']}"].append((r["f_act"], r["f_beh"]))
    out: dict[str, dict] = {}
    for k, pts in sorted(per.items()):
        acts = [a for a, _ in pts]
        behs = [b for _, b in pts]
        out[k] = {
            "act_mean": float(np.mean(acts)),
            "beh_mean": float(np.mean(behs)),
            "n_rows": len(pts),
            "in_rho": len(pts) >= 2 and max(acts) > min(acts) and max(behs) > min(behs),
        }
    return out


def fig_act_beh_agreement(
    arm_rows: dict[str, list[dict]], out_dir: Path, inputs: list[Path]
) -> None:
    """Manifest ``act_beh_agreement``: mean F_act (read layer 26, disjoint
    floor halves) vs mean F_beh per (cell x slot x arm), separation-excluded.
    Per-arm Spearman rho is computed ONLY across units passing the
    rule-19-mirrored dynamic-range screen (``ACT_BEH_DYNAMIC_RANGE_SCREEN``),
    with the screened points' realized dynamic range stated in-panel; units
    failing the screen render hollow and carry no rho weight (r3 MINOR 1)."""
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(9, 8))
    panel_lines = ["rho over units with dynamic range (rule-19-mirrored screen):"]
    dropped_any = False
    for arm, rows in arm_rows.items():
        units = _act_beh_units(rows)
        kept = {k: u for k, u in units.items() if u["in_rho"]}
        dropped = {k: u for k, u in units.items() if not u["in_rho"]}
        xs = [u["act_mean"] for u in kept.values()]
        ys = [u["beh_mean"] for u in kept.values()]
        rho = spearmanr(xs, ys)[0] if len(xs) >= 5 else float("nan")
        ax.scatter(
            xs,
            ys,
            s=18,
            color=ARM_COLORS[arm],
            alpha=0.8,
            label=f"{arm} (rho={rho:.3f} over n={len(kept)} screened; {len(dropped)} dropped)",
        )
        if dropped:
            dropped_any = True
            ax.scatter(
                [u["act_mean"] for u in dropped.values()],
                [u["beh_mean"] for u in dropped.values()],
                s=18,
                facecolors="none",
                edgecolors=ARM_COLORS[arm],
                alpha=0.8,
            )
        for k, u in units.items():
            ax.annotate(k, (u["act_mean"], u["beh_mean"]), fontsize=3.5, alpha=0.6)
        if kept:
            panel_lines.append(
                f"{arm}: F_act range [{min(xs):+.3f}, {max(xs):+.3f}], "
                f"F_beh range [{min(ys):+.3f}, {max(ys):+.3f}] (n={len(kept)})"
            )
    if dropped_any:
        panel_lines.append("hollow points: screened out of rho (no dynamic range)")
    ax.text(
        0.02,
        0.98,
        "\n".join(panel_lines),
        transform=ax.transAxes,
        fontsize=5.5,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.7, "lw": 0.3},
    )
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.5)
    ax.set_xlabel("mean F_act (read layer 26)")
    ax.set_ylabel("mean F_beh")
    ax.set_title("F_act vs F_beh per (cell x slot x arm), separation-excluded")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "act_beh_agreement", inputs)


def fig_crosstype_by_donor(
    stats: dict,
    null_crosstype: list[dict],
    null_shuffled: list[dict],
    out_dir: Path,
    inputs: list[Path],
) -> None:
    """Manifest ``crosstype_null_by_donor``: (i) donor-TYPE-resolved cross-type
    null F_beh per elevated recipient cell (pooled crosstype 95% CI excludes
    0); (ii) donor-VALUE-resolved shuffled null for the two ordinal value sets
    (refusal_boundary, constraint_knowledge) when either null's CI excludes 0."""
    from explore_persona_space.experiments.issue2162 import bank2162 as B

    per_cell = stats["per_cell"]

    def _ci_excludes_zero(rec: dict, arm: str) -> bool:
        ci = (rec.get("ci95") or {}).get(arm, [None, None])
        return ci[0] is not None and ci[1] is not None and (ci[0] > 0 or ci[1] < 0)

    kept_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in null_crosstype:
        if r["f_beh"] is not None and r["separation"] is not None and abs(r["separation"]) >= 0.5:
            kept_rows[(r["cell"], r["slot"])].append(r)
    elevated = [key for key, rec in sorted(per_cell.items()) if _ci_excludes_zero(rec, "crosstype")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax = axes[0]
    x = 0.0
    ticks, tick_labels = [], []
    for key in elevated:
        cell, slot = key.split("|")
        by_donor: dict[str, list[float]] = defaultdict(list)
        for r in kept_rows.get((cell, slot), []):
            donor = B.base_type_of(r["donor_cell"]) if r.get("donor_cell") else "?"
            by_donor[donor].append(r["f_beh"])
        for j, donor in enumerate(sorted(by_donor)):
            vals = by_donor[donor]
            ax.bar(x + j * 0.6, float(np.mean(vals)), 0.5, color="#c44e52", alpha=0.75)
            ax.annotate(
                f"{donor} (n={len(vals)})",
                (x + j * 0.6, float(np.mean(vals))),
                fontsize=4.5,
                rotation=90,
                ha="center",
            )
        ticks.append(x + max(len(by_donor) - 1, 0) * 0.3)
        tick_labels.append(key)
        x += max(len(by_donor), 1) * 0.6 + 0.8
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=6, ha="right")
    ax.set_ylabel("crosstype-null F_beh (mean per donor type)")
    ax.set_title(
        f"Cross-type null split by donor type — {len(elevated)} recipient cells "
        "whose pooled crosstype 95% CI excludes 0" + (" (none)" if not elevated else "")
    )
    ax = axes[1]
    ordinal = ("refusal_boundary", "constraint_knowledge")
    shuf_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in null_shuffled:
        if (
            r["f_beh"] is not None
            and r["separation"] is not None
            and abs(r["separation"]) >= 0.5
            and B.base_type_of(r["cell"]) in ordinal
        ):
            shuf_rows[(r["cell"], r["slot"])].append(r)
    shown = [
        key
        for key in sorted(f"{c}|{s}" for c, s in shuf_rows)
        if (rec := per_cell.get(key)) is not None
        and (_ci_excludes_zero(rec, "shuffled") or _ci_excludes_zero(rec, "crosstype"))
    ]
    x = 0.0
    ticks, tick_labels = [], []
    for key in shown:
        cell, slot = key.split("|")
        by_val: dict[str, list[float]] = defaultdict(list)
        for r in shuf_rows.get((cell, slot), []):
            by_val[str(r.get("donor_value_b"))].append(r["f_beh"])
        for j, val in enumerate(sorted(by_val)):
            vals = by_val[val]
            ax.bar(x + j * 0.6, float(np.mean(vals)), 0.5, color="#9d9d9d", alpha=0.85)
            ax.annotate(
                f"{val} (n={len(vals)})",
                (x + j * 0.6, float(np.mean(vals))),
                fontsize=4.5,
                rotation=90,
                ha="center",
            )
        ticks.append(x + max(len(by_val) - 1, 0) * 0.3)
        tick_labels.append(key)
        x += max(len(by_val), 1) * 0.6 + 0.8
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=6, ha="right")
    ax.set_ylabel("shuffled-null F_beh (mean per donor value)")
    ax.set_title(
        "Shuffled null split by donor value — ordinal value sets "
        "(refusal_boundary, constraint_knowledge) with either null CI excluding 0"
        + (" (none)" if not shown else "")
    )
    _save(fig, out_dir, "crosstype_null_by_donor", inputs)


def fig_stage2_layer_profile(stage2_cells: list[dict], out_dir: Path, inputs: list[Path]) -> None:
    """Manifest ``layer_profile_stage2`` (+ ``_perpair``): survivor x layer
    heatmaps of stage-2 steered F_beh per dose, with the steered-minus-
    shuffled companion panel; per-pair points at each survivor's best
    (layer, dose) with steered and shuffled interleaved, pair-id labeled."""
    kept = [
        r
        for r in stage2_cells
        if r["f_beh"] is not None and r["separation"] is not None and abs(r["separation"]) >= 0.5
    ]
    assert kept, "stage2_cells has no separation-kept scored rows"
    doses = sorted({r["dose"] for r in kept})
    layers = sorted({r["layer"] for r in kept})
    units = sorted({f"{r['cell']}|{r['slot']}" for r in kept})
    mean_f: dict[tuple[str, str, int, int], float] = {}
    per_pair: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for r in kept:
        per_pair[(f"{r['cell']}|{r['slot']}", r["arm"], r["layer"], r["dose"])].append(r)
    for key, rows in per_pair.items():
        mean_f[key] = float(np.mean([r["f_beh"] for r in rows]))

    fig, axes = plt.subplots(2, len(doses), figsize=(7 * len(doses), 9), sharey=True, squeeze=False)
    for di, dose in enumerate(doses):
        steered = np.full((len(units), len(layers)), np.nan)
        delta = np.full((len(units), len(layers)), np.nan)
        for ui, unit in enumerate(units):
            for li, layer in enumerate(layers):
                s = mean_f.get((unit, "steered", layer, dose))
                sh = mean_f.get((unit, "shuffled", layer, dose))
                if s is not None:
                    steered[ui, li] = s
                if s is not None and sh is not None:
                    delta[ui, li] = s - sh
        for row_i, (mat, lab, cmap) in enumerate(
            ((steered, "steered F_beh", "viridis"), (delta, "steered - shuffled", "coolwarm"))
        ):
            ax = axes[row_i][di]
            im = ax.imshow(mat, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, fontsize=7)
            ax.set_yticks(range(len(units)))
            ax.set_yticklabels(units, fontsize=6)
            ax.set_xlabel("layer")
            ax.set_title(f"dose {dose}: {lab} (post-selection survivors)")
            fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out_dir, "layer_profile_stage2", inputs)

    # Per-pair companion at each survivor's best steered (layer, dose).
    fig, ax = plt.subplots(figsize=(max(10, len(units) * 0.9), 6))
    rng = np.random.default_rng(2162)
    for ui, unit in enumerate(units):
        best = None
        for layer in layers:
            for dose in doses:
                v = mean_f.get((unit, "steered", layer, dose))
                if v is not None and (best is None or v > best[0]):
                    best = (v, layer, dose)
        if best is None:
            continue
        _, layer, dose = best
        for arm, off, color in (("steered", -0.12, "#4878d0"), ("shuffled", 0.12, "#9d9d9d")):
            for r in per_pair.get((unit, arm, layer, dose), []):
                xi = ui + off + float(rng.uniform(-0.05, 0.05))
                ax.scatter(xi, r["f_beh"], s=9, color=color, alpha=0.7)
                ax.annotate(r["pair_id"], (xi, r["f_beh"]), fontsize=3.2, alpha=0.7)
        ax.text(
            ui,
            1.02,
            f"L{layer},d{dose}",
            ha="center",
            fontsize=5.5,
            transform=ax.get_xaxis_transform(),
        )
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(units)))
    ax.set_xticklabels(units, rotation=45, fontsize=6, ha="right")
    ax.set_ylabel("per-pair stage-2 F_beh")
    ax.set_title(
        "Stage-2 per-pair F_beh at each survivor's best (layer, dose) — "
        "steered (blue) vs shuffled null (grey), pair-id labeled"
    )
    _save(fig, out_dir, "layer_profile_stage2_perpair", inputs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2162 figure set.")
    ap.add_argument("--metrics-dir", type=Path, default=Path("eval_results/issue_2162/f_metrics"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/issue_2162"))
    ap.add_argument("--only", default=None, help="comma-separated figure subset (manifest keys)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    md = args.metrics_dir
    stats = json.loads((md / "stats.json").read_text())
    f_cells = list(_iter_jsonl(md / "f_cells.jsonl"))
    null_shuffled = list(_iter_jsonl(md / "null_shuffled_cells.jsonl"))
    null_crosstype = list(_iter_jsonl(md / "null_crosstype_cells.jsonl"))
    anchor_rows = list(_iter_jsonl(md / "anchors.jsonl"))
    arm_rows = {"steered": f_cells, "shuffled": null_shuffled, "crosstype": null_crosstype}
    manifest = {
        "hero": lambda: fig_hero(
            stats,
            arm_rows,
            args.out_dir,
            [
                md / "stats.json",
                md / "f_cells.jsonl",
                md / "null_shuffled_cells.jsonl",
                md / "null_crosstype_cells.jsonl",
            ],
        ),
        "two_by_two": lambda: fig_two_by_two(
            json.loads((md / "two_by_two.json").read_text()),
            args.out_dir,
            [md / "two_by_two.json"],
        ),
        "layer_profile": lambda: fig_layer_profile(
            json.loads((md / "probe.json").read_text()),
            md / "probe_perm_matrix" / "perm_auc_matrix.npz",
            args.out_dir,
            [md / "probe.json", md / "probe_perm_matrix" / "perm_auc_matrix.npz"],
        ),
        "route_contrasts": lambda: fig_route_contrasts(stats, args.out_dir, [md / "stats.json"]),
        "route_contrasts_perpair": lambda: fig_route_contrasts_perpair(
            f_cells, args.out_dir, [md / "f_cells.jsonl"]
        ),
        "dose_position": lambda: fig_dose_position(stats, args.out_dir, [md / "stats.json"]),
        "recency_load_perpair": lambda: fig_recency_load_perpair(
            f_cells, args.out_dir, [md / "f_cells.jsonl"]
        ),
        "anchor_separation": lambda: fig_anchor_separation(
            anchor_rows, args.out_dir, [md / "anchors.jsonl"]
        ),
        "act_beh_agreement": lambda: fig_act_beh_agreement(
            arm_rows,
            args.out_dir,
            [
                md / "f_cells.jsonl",
                md / "null_shuffled_cells.jsonl",
                md / "null_crosstype_cells.jsonl",
            ],
        ),
        "crosstype_by_donor": lambda: fig_crosstype_by_donor(
            stats,
            null_crosstype,
            null_shuffled,
            args.out_dir,
            [
                md / "stats.json",
                md / "null_crosstype_cells.jsonl",
                md / "null_shuffled_cells.jsonl",
            ],
        ),
        "margin_validation": lambda: fig_margin_validation(
            list(_iter_jsonl(md / "margin_cells.jsonl")),
            f_cells,
            json.loads((md / "margin_validation.json").read_text()),
            args.out_dir,
            [md / "margin_cells.jsonl", md / "margin_validation.json"],
        ),
        "stage2_layer_profile": lambda: fig_stage2_layer_profile(
            list(_iter_jsonl(md / "stage2_cells.jsonl")),
            args.out_dir,
            [md / "stage2_cells.jsonl"],
        ),
        "diagnostics": lambda: fig_diagnostics(
            stats,
            arm_rows,
            anchor_rows,
            args.out_dir,
            [md / "f_cells.jsonl", md / "anchors.jsonl"],
        ),
    }
    only = set(args.only.split(",")) if args.only else set(manifest)
    unknown = only - set(manifest)
    assert not unknown, f"unknown figure keys: {sorted(unknown)}"
    # Later-phase inputs (margin, stage-2) may legitimately not exist yet: a
    # default all-figures run drops those keys with a LOUD log line; a figure
    # requested EXPLICITLY via --only still hard-fails on its missing input.
    later_phase = {
        "margin_validation": md / "margin_cells.jsonl",
        "stage2_layer_profile": md / "stage2_cells.jsonl",
    }
    if not args.only:
        for name, path in later_phase.items():
            if name in only and not path.exists():
                logger.info("[figures] SKIP %s — missing later-phase input %s", name, path)
                only.discard(name)
    for name in manifest:
        if name in only:
            logger.info("[figures] building %s", name)
            manifest[name]()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
