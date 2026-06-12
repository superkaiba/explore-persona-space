#!/usr/bin/env python3
"""#602 figures — hero 1 (estimator-vs-realized matrix), hero 2 (repair
scatter), exploratory dump (per-layer curves, K-sweep, position deltas).

Reads the Phase-2 JSONs (``eval_results/issue_602/{agreement,repair,grids}``)
and renders to ``figures/issue_602/`` with the paper-plots conventions
(colorblind-safe palette, commit-pinned meta.json, PNG+PDF).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis import i602_bakeoff as bk  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    set_paper_style,
)

logger = logging.getLogger("issue602_figures")

FAMILY_LABELS = {
    "marker519": "Marker glyph (519)",
    "em_turner": "Bad medical advice (turner)",
    "fact541": "Planted fact (541)",
    "refusal518": "Refusal (518)",
    "em518": "Contrastive EM (518)",
    "loc474": "Marker, localized arm (474)",
}
EST_LABELS = {
    "est_tf": "Replay training answers (E1)",
    "est_icl": "Examples in prompt (E2)",
    "est_desc": "Describe in words (E3)",
}
# Reader-facing source descriptors (no project slugs in rendered labels).
SOURCE_LABELS = {
    "assistant": "assistant persona",
    "comedian": "comedian persona",
    "kindergarten_teacher": "kindergarten teacher",
    "qwen_default": "default (no persona)",
    "software_engineer": "software engineer",
    "villain": "villain persona",
    "no_system": "no system prompt",
    "medical_doctor": "medical doctor",
    "marine_biologist": "marine biologist",
    "courthouse_architecture_historian": "courthouse historian",
    "wooden_furniture_carpenter": "furniture carpenter",
    # #474 transformation contexts (i406 condition ids decoded)
    "A1": "'helpful assistant' context",
    "B1": "bare-question context",
    "C1": "standard Qwen template",
    "D1": "formal-register rewrite",
}
FAMILY_SHORT = {
    "marker519": "Marker glyph",
    "em_turner": "Bad medical advice",
    "fact541": "Planted fact",
    "refusal518": "Refusal",
    "em518": "Contrastive EM",
    "loc474": "Marker, localized arm",
}


def unit_label(unit: str) -> str:
    """Plain-English label for an estimator unit key like ``em518__assistant``."""
    fam, _, src = unit.partition("__")
    return f"{FAMILY_SHORT.get(fam, fam)} — {SOURCE_LABELS.get(src, src)}"


def _save(fig, out_dir: Path, name: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, default=float))
    plt.close(fig)
    logger.info("wrote %s/%s.{png,pdf,meta.json}", out_dir, name)


def _meta(headline: dict, name: str) -> dict:
    return {
        "issue": bk.ISSUE,
        "figure": name,
        "git_commit": bk.git_sha(REPO),
        "source_git_commit": headline.get("reproducibility", {}).get("git_commit"),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def hero1_matrix(
    headline: dict,
    out_dir: Path,
    name: str = "hero1_estimator_validity_matrix",
    construction_label: str = (
        "at the pre-registered construction (L14, mean-response, variant-matched base)"
    ),
) -> None:
    """Hero 1: per-(estimator x family) median cosines, both targets, with
    the random-null band, sibling-excluded off-diag distribution, and
    cross-seed ceiling whiskers. ``name``/``construction_label`` let the
    layer-re-read companion (l27_verdict_matrix) reuse the exact renderer."""
    families = [f for f in bk.FAMILIES]
    estimators = ["est_tf", "est_icl", "est_desc"]
    verd = headline["verdicts"]
    null95 = headline["null_random"]["p95"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    width = 0.25
    x = np.arange(len(families))
    colors = paper_palette(3)
    for ax, target_key, title in (
        (axes[0], "median_cos_w_shared", "vs shared direction (w_shared, primary)"),
        (axes[1], "median_cos_w_src", "vs source-context write (w_src, secondary)"),
    ):
        for ei, estimator in enumerate(estimators):
            vals, offdiag = [], []
            for fam in families:
                v = verd.get(f"{fam}__{estimator}", {})
                vals.append(v.get(target_key) if v.get("verdict") != "MISSING" else np.nan)
                m = v.get("margin_excl_siblings")
                offdiag.append(
                    (v.get(target_key) - m)
                    if (m is not None and v.get(target_key) is not None)
                    else np.nan
                )
            ax.bar(
                x + (ei - 1) * width,
                vals,
                width,
                label=EST_LABELS[estimator],
                color=colors[ei],
            )
            ax.scatter(
                x + (ei - 1) * width,
                offdiag,
                marker="_",
                s=180,
                color="black",
                zorder=5,
                linewidths=1.4,
            )
        # ceiling whiskers (cross-seed U1 cosine) where available
        for fi, fam in enumerate(families):
            ceil_vals = [
                c["median_abs_cos"]
                for key, c in headline.get("ceiling_seed", {}).items()
                if key.startswith(f"{fam}__")
            ]
            if ceil_vals:
                ax.plot(
                    [fi - 0.38, fi + 0.38],
                    [float(np.median(ceil_vals))] * 2,
                    color=paper_palette_role("neutral"),
                    linestyle=":",
                    linewidth=1.4,
                )
        ax.axhspan(-null95, null95, color="gray", alpha=0.18)
        ax.axhline(0.3, color=paper_palette_role("accent"), linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in families], rotation=28, ha="right")
        ax.set_title(title)
    axes[0].set_ylabel("median cos(estimated write, realized write)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Estimator validity {construction_label} — gray band = 10k random null, "
        "dashed = 0.3 validity bar, dotted = cross-seed ceiling, "
        "black ticks = sibling-excluded off-diagonal mean",
        fontsize=9,
    )
    _save(fig, out_dir, name, _meta(headline, name))


def hero2_repair(
    repair: dict,
    out_dir: Path,
    name: str = "hero2_repair_scatter",
    title_note: str = "",
) -> None:
    """Hero 2: repair scatter — estimator behavioral rho vs realized
    behavioral rho per cell, quadrants labeled, norm-only overplotted.
    ``name``/``title_note`` let the layer-re-read companion
    (l27_repair_scatter) reuse the exact renderer."""
    rows = repair["repair_rows"]
    if not rows:
        logger.warning("no repair rows — skipping hero 2")
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fams = sorted({r["family"] for r in rows})
    colors = dict(zip(fams, paper_palette(max(len(fams), 3)), strict=False))
    for r in rows:
        ax.scatter(
            r["rho_behav_est"],
            r["rho_behav_real"],
            color=colors[r["family"]],
            s=44,
            edgecolor="black",
            linewidth=0.4,
            zorder=4,
        )
        ax.scatter(
            r["rho_behav_norm"],
            r["rho_behav_real"],
            facecolors="none",
            edgecolors=colors[r["family"]],
            s=52,
            marker="s",
            linewidths=1.3,  # style zeroes default linewidths -> invisible open markers
            zorder=3,
        )
    ax.axvline(0.3, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("estimator behavioral Spearman rho")
    ax.set_ylabel("realized-write behavioral Spearman rho")
    label_bbox = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.6)
    ax.text(
        0.02,
        0.90,
        "repair-positive",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=label_bbox,
        zorder=6,
    )
    ax.text(
        0.98,
        0.90,
        "both-pass",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="right",
        bbox=label_bbox,
        zorder=6,
    )
    ax.text(
        0.02,
        0.03,
        "both-fail (update rule implicated)",
        transform=ax.transAxes,
        fontsize=8,
        bbox=label_bbox,
        zorder=6,
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[f], label=FAMILY_LABELS[f])
        for f in fams
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="",
            markerfacecolor="none",
            markeredgewidth=1.3,  # match the visible open-square overlay
            color="black",
            label="norm-only predictor (direction-free)",
        )
    )
    ax.legend(handles=handles, fontsize=7.5, loc="lower right")
    ax.set_title(
        "Failure localization: does substituting the realized write repair the ranking?"
        + title_note
    )
    _save(fig, out_dir, name, _meta(repair, name))


def exploratory_dump(grid: dict, headline: dict, out_dir: Path) -> None:
    """Per-layer agreement curves, K-sweep lines, position paired deltas."""
    rows = grid["rows"]
    if not rows:
        logger.warning("empty exploratory grid — skipping dump")
        return
    # per-layer curves at mean_resp, per estimator (median over cells)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fams = sorted({r["cell_id"].split("__")[0] for r in rows})
    colors = dict(zip(fams, paper_palette(max(len(fams), 3)), strict=False))
    for ax, estimator in zip(axes, ("est_tf", "est_icl", "est_desc"), strict=True):
        for fam in fams:
            xs, ys = [], []
            for ly in bk.LAYERS:
                vals = [
                    r["cos_w_shared"]
                    for r in rows
                    if r["estimator"] == estimator
                    and r["layer"] == ly
                    and r["pos"] == "mean_resp"
                    and r["cell_id"].startswith(fam)
                    and (r["K"] in (None, bk.E2_K_PRIMARY))
                    and r.get("e1_read") != "incl_marker"
                ]
                if vals:
                    xs.append(ly)
                    ys.append(float(np.median(vals)))
            if xs:
                ax.plot(xs, ys, marker="o", label=FAMILY_LABELS.get(fam, fam), color=colors[fam])
        ax.set_title(EST_LABELS[estimator], fontsize=9)
        ax.set_xlabel("layer")
        ax.axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("median cos(w_hat, w_shared)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Exploratory: per-layer agreement (mean-response read)", fontsize=10)
    _save(fig, out_dir, "exp_per_layer_agreement", _meta(headline, "exp_layers"))

    # K-sweep (est_icl only)
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam in fams:
        xs, ys = [], []
        for k in bk.E2_K_SWEEP:
            vals = [
                r["cos_w_shared"]
                for r in rows
                if r["estimator"] == "est_icl"
                and r["K"] == k
                and r["layer"] == bk.PRIMARY_LAYER
                and r["pos"] == "mean_resp"
                and r["cell_id"].startswith(fam)
            ]
            if vals:
                xs.append(k)
                ys.append(float(np.median(vals)))
        if xs:
            ax.plot(xs, ys, marker="o", label=FAMILY_LABELS.get(fam, fam), color=colors[fam])
    ax.set_xlabel("K (in-context demonstrations)")
    ax.set_ylabel("median cos(w_hat_E2, w_shared) @ L14/mean-resp")
    ax.legend(fontsize=7)
    ax.set_title("Exploratory: ICL dose sweep")
    _save(fig, out_dir, "exp_k_sweep", _meta(headline, "exp_k"))

    # position paired deltas (mean_resp vs slot/last-tok at L14)
    fig, ax = plt.subplots(figsize=(6, 4))
    pts = []
    for r in rows:
        if (
            r["layer"] != bk.PRIMARY_LAYER
            or r["pos"] != "mean_resp"
            or r.get("e1_read") == "incl_marker"
        ):
            continue
        twin = next(
            (
                t
                for t in rows
                if t["cell_id"] == r["cell_id"]
                and t["estimator"] == r["estimator"]
                and t["K"] == r["K"]
                and t["layer"] == r["layer"]
                and t["pos"] == "slot"
            ),
            None,
        )
        if twin:
            pts.append((r["cos_w_shared"], twin["cos_w_shared"]))
    if pts:
        arr = np.array(pts)
        ax.scatter(arr[:, 0], arr[:, 1], s=20, alpha=0.7, color=paper_palette_role("primary"))
        lim = max(0.05, float(np.abs(arr).max()) * 1.1)
        ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.8)
        ax.set_xlabel("cos @ mean-response read")
        ax.set_ylabel("cos @ slot / last-token read")
        ax.set_title("Exploratory: read-position paired deltas (L14)")
        _save(fig, out_dir, "exp_position_deltas", _meta(headline, "exp_pos"))


def exp_cross_estimator(headline: dict, out_dir: Path) -> None:
    """H5 diagnostics: within-unit pairwise estimator agreement heatmap +
    SAME-estimator cross-family cosine distributions (the generic-
    prompting-attractor diagnostic)."""
    ce = headline.get("cross_estimator")
    if not ce or not ce.get("pairwise_within_unit"):
        logger.warning("no cross_estimator block — skipping")
        return
    pairwise = ce["pairwise_within_unit"]
    pair_keys = ["est_tf__vs__est_icl", "est_tf__vs__est_desc", "est_icl__vs__est_desc"]
    units = sorted(pairwise.keys())
    mat = np.full((len(units), len(pair_keys)), np.nan)
    for ui, u in enumerate(units):
        for pi, pk in enumerate(pair_keys):
            if pk in pairwise[u]:
                mat[ui, pi] = pairwise[u][pk]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3.6, 0.34 * len(units))), width_ratios=[1.1, 1])
    im = axes[0].imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(range(len(pair_keys)))
    axes[0].set_xticklabels(["E1 vs E2", "E1 vs E3", "E2 vs E3"])
    axes[0].set_yticks(range(len(units)))
    axes[0].set_yticklabels([unit_label(u) for u in units], fontsize=6.5)
    for ui in range(len(units)):
        for pi in range(len(pair_keys)):
            if not np.isnan(mat[ui, pi]):
                axes[0].text(pi, ui, f"{mat[ui, pi]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    axes[0].set_title("Within-unit pairwise estimator agreement (L14, mean-resp)", fontsize=9)
    cross_fam = ce.get("same_estimator_cross_family", {})
    data, labels = [], []
    for estimator in ("est_tf", "est_icl", "est_desc"):
        vals = list(cross_fam.get(estimator, {}).values())
        if vals:
            data.append(vals)
            labels.append(EST_LABELS[estimator])
    if data:
        axes[1].boxplot(data, tick_labels=labels)
        for i, vals in enumerate(data):
            axes[1].scatter(
                np.full(len(vals), i + 1)
                + np.random.default_rng(0).uniform(-0.06, 0.06, len(vals)),
                vals,
                s=12,
                alpha=0.6,
                color=paper_palette(3)[i],
            )
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].set_ylabel("cos(same estimator, different family)")
        axes[1].tick_params(axis="x", labelsize=7)
        axes[1].set_title(
            "Cross-family same-estimator cosines\n(prompting-attractor check)", fontsize=9
        )
    _save(fig, out_dir, "exp_cross_estimator", _meta(headline, "exp_cross_estimator"))


def exp_projection_scatter(repair: dict, out_dir: Path) -> None:
    """Per-context projection scatter, RAW alongside RANKED (plan §6):
    estimator projection vs LOCO realized projection per context."""
    geo = repair.get("geometry_consistency_rows", [])
    geo = [g for g in geo if g.get("profiles")]
    if not geo:
        logger.warning("no geometry rows with profiles — skipping projection scatter")
        return
    fams = sorted({g["cell_id"].split("__")[0] for g in geo})
    colors = dict(zip(fams, paper_palette(max(len(fams), 3)), strict=False))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=False)
    for ci, estimator in enumerate(("est_tf", "est_icl", "est_desc")):
        raw_ax, rank_ax = axes[0][ci], axes[1][ci]
        for g in geo:
            if g["estimator"] != estimator:
                continue
            fam = g["cell_id"].split("__")[0]
            prof = g["profiles"]
            ctxs = sorted(prof.keys())
            est_vals = np.array([prof[c]["est"] for c in ctxs])
            real_vals = np.array([prof[c]["real"] for c in ctxs])
            raw_ax.scatter(est_vals, real_vals, s=14, alpha=0.6, color=colors[fam])
            rank_ax.scatter(
                est_vals.argsort().argsort(),
                real_vals.argsort().argsort(),
                s=14,
                alpha=0.6,
                color=colors[fam],
            )
        raw_ax.set_title(EST_LABELS[estimator], fontsize=9)
        raw_ax.set_xlabel("dv(c) @ unit(w_hat)  [raw]")
        rank_ax.set_xlabel("rank of estimator projection")
    axes[0][0].set_ylabel("dv(c) @ unit(w_shared LOCO)  [raw]")
    axes[1][0].set_ylabel("rank of realized projection")
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colors[f], label=FAMILY_LABELS.get(f, f))
        for f in fams
    ]
    axes[0][0].legend(handles=handles, fontsize=6.5)
    fig.suptitle(
        "Exploratory: per-context projections, raw (top) alongside ranked (bottom)", fontsize=10
    )
    _save(fig, out_dir, "exp_projection_scatter", _meta(repair, "exp_projection_scatter"))


def exp_reliability(grid: dict, headline: dict, out_dir: Path) -> None:
    """Split-half reliabilities (all three estimators) + E1 subsample
    stability curves, from the persisted per-row stacks."""
    rel = grid.get("reliability", {})
    if not rel:
        logger.warning("no reliability block — skipping")
        return
    bars: list[tuple[str, str, float]] = []  # (unit, est_class, split_half_cos)
    curves: dict[str, list[tuple[float, float]]] = {}
    for unit, entries in sorted(rel.items()):
        for key, e in entries.items():
            if "split_half_cos" not in e:
                continue
            est_class = "E1" if key.startswith("e1") else ("E2" if key.startswith("e2") else "E3")
            bars.append((f"{unit}::{key}", est_class, e["split_half_cos"]))
            if key.startswith("e1") and "subsample_stability_vs_full" in e:
                pts = [
                    (float(fk.removeprefix("frac_")), sv["mean_cos_vs_full"])
                    for fk, sv in e["subsample_stability_vs_full"].items()
                ]
                curves[f"{unit}::{key}"] = sorted(pts)
    if not bars:
        logger.warning("reliability block empty — skipping figure")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.6, 0.22 * len(bars))))
    cls_colors = dict(zip(("E1", "E2", "E3"), paper_palette(3), strict=True))
    ys = np.arange(len(bars))
    axes[0].barh(ys, [b[2] for b in bars], color=[cls_colors[b[1]] for b in bars])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([b[0] for b in bars], fontsize=5.5)
    axes[0].axvline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("split-half cos(w_hat_half1, w_hat_half2) @ L14/mean-resp")
    axes[0].set_title("Split-half reliability per estimator unit", fontsize=9)
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=c, label=k)
        for k, c in cls_colors.items()
    ]
    axes[0].legend(handles=handles, fontsize=7)
    for _label, pts in curves.items():
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=1, alpha=0.7)
    axes[1].set_xlabel("E1 row fraction")
    axes[1].set_ylabel("mean cos vs full-row w_hat")
    axes[1].set_title("E1 row-count subsample stability", fontsize=9)
    _save(fig, out_dir, "exp_reliability", _meta(headline, "exp_reliability"))


def exp_e1_marker_reads(grid: dict, headline: dict, out_dir: Path) -> None:
    """Include- vs exclude-marker E1 reads (marker families, L14/mean-resp)."""
    rows = grid.get("rows", [])
    excl = {
        (r["cell_id"], r["layer"]): r["cos_w_shared"]
        for r in rows
        if r["estimator"] == "est_tf" and r.get("e1_read") == "excl_marker"
    }
    incl = {
        (r["cell_id"], r["layer"]): r["cos_w_shared"]
        for r in rows
        if r["estimator"] == "est_tf" and r.get("e1_read") == "incl_marker"
    }
    common = sorted(set(excl) & set(incl))
    if not common:
        logger.warning("no paired include/exclude-marker E1 rows — skipping")
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    xs = [excl[k] for k in common]
    ys = [incl[k] for k in common]
    ax.scatter(xs, ys, s=22, alpha=0.75, color=paper_palette_role("primary"))
    lim = max(0.05, float(np.abs(np.array(xs + ys)).max()) * 1.15)
    ax.plot([-lim, lim], [-lim, lim], color="gray", linewidth=0.8)
    ax.set_xlabel("cos(w_hat_E1 exclude-marker, w_shared)")
    ax.set_ylabel("cos(w_hat_E1 include-marker, w_shared)")
    ax.set_title(
        "E1 marker-family reads: token-identity component\n"
        "(include vs exclude the trailing marker)",
        fontsize=9,
    )
    _save(fig, out_dir, "exp_e1_marker_reads", _meta(headline, "exp_e1_marker_reads"))


def shuffle_control_bars(sc: dict, out_dir: Path, name: str = "shuffle_control_bars") -> None:
    """Follow-up figure (plan v3 §3.5): per-cell grouped bars — intact /
    shuffle (matched) / mismatch cos to w_shared at L27/mean-resp — with
    the 0.3 collapse bar and the random-null p95 band; the UNMATCHED
    shuffle contrast overplotted as open markers (sensitivity, never
    gates)."""
    rows = [r for r in sc["per_cell"] if not r.get("missing")]
    cell_ids = sorted({r["cell_id"] for r in rows})
    if not cell_ids:
        logger.warning("no shuffle-control rows — skipping figure")
        return
    by = {(r["cell_id"], r["transform"]): r for r in rows}

    def _cell_label(cid: str) -> str:
        fam, src, seed = cid.split("__")
        return f"{FAMILY_SHORT.get(fam, fam)}\n{SOURCE_LABELS.get(src, src)} ({seed})"

    transforms = ["intact", "shuffle", "mismatch"]
    t_labels = {
        "intact": "intact replay (control)",
        "shuffle": "within-completion shuffle (matched contrast)",
        "mismatch": "question-mismatched pairing",
    }
    null95 = sc["null_random"]["p95"]
    x = np.arange(len(cell_ids))
    width = 0.26
    colors = paper_palette(3)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.35 * len(cell_ids)), 5.0))
    for ti, t in enumerate(transforms):
        vals = [by.get((cid, t), {}).get("cos_w_shared") for cid in cell_ids]
        ax.bar(
            x + (ti - 1) * width,
            [np.nan if v is None else v for v in vals],
            width,
            label=t_labels[t],
            color=colors[ti],
        )
    un_vals = [by.get((cid, "shuffle_unmatched"), {}).get("cos_w_shared") for cid in cell_ids]
    ax.scatter(
        x,
        [np.nan if v is None else v for v in un_vals],
        facecolors="none",
        edgecolors="black",
        s=46,
        marker="o",
        linewidths=1.3,
        zorder=5,
        label="shuffle, unmatched contrast (sensitivity)",
    )
    ax.axhspan(-null95, null95, color="gray", alpha=0.18)
    ax.axhline(0.3, color=paper_palette_role("accent"), linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([_cell_label(c) for c in cell_ids], fontsize=7)
    ax.set_ylabel("cos(estimated write, shared realized direction)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title(
        "Token-integrity control at the layer-27 read — does replay validity survive "
        "destroying token order?\n(gray band = 10k random null, dashed = 0.3 collapse bar)",
        fontsize=9,
    )
    _save(fig, out_dir, name, _meta(sc, name))


def main() -> int:
    """Render all #602 figures from the Phase-2 JSONs."""
    parser = argparse.ArgumentParser(description="#602 figures")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--shuffle-control",
        action="store_true",
        help=(
            "Render ONLY the shuffled-replay-l27-control follow-up figure from "
            "shuffled-replay-l27-control/shuffle_control.json"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    set_paper_style()
    ev = Path(args.eval_dir) if args.eval_dir else bk.eval_dir(REPO)
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "figures" / "issue_602"
    if args.shuffle_control:
        sc = json.loads((ev / bk.FOLLOWUP_SHUFFLE_SLUG / "shuffle_control.json").read_text())
        shuffle_control_bars(sc, out_dir)
        logger.info("shuffle-control figure complete")
        return 0
    headline = json.loads((ev / "agreement" / "headline_metrics.json").read_text())
    repair = json.loads((ev / "repair" / "repair_test.json").read_text())
    grid = json.loads((ev / "grids" / "exploratory_grid.json").read_text())
    hero1_matrix(headline, out_dir)
    hero2_repair(repair, out_dir)
    exploratory_dump(grid, headline, out_dir)
    exp_cross_estimator(headline, out_dir)
    exp_projection_scatter(repair, out_dir)
    exp_reliability(grid, headline, out_dir)
    exp_e1_marker_reads(grid, headline, out_dir)
    # layer-re-read companions (free-analysis follow-up 1): render a verdict
    # matrix + repair scatter per l{N}_reread.json found, same renderers.
    for reread_path in sorted((ev / "agreement").glob("l*_reread.json")):
        reread = json.loads(reread_path.read_text())
        ly = reread["layer"]
        repair_l_path = ev / "repair" / f"repair_test_l{ly}.json"
        hero1_matrix(
            reread,
            out_dir,
            name=f"l{ly}_verdict_matrix",
            construction_label=(
                f"(L{ly}, mean-response, variant-matched base — exploratory re-read; "
                "the committed verdict is the L14 read)"
            ),
        )
        if repair_l_path.exists():
            repair_l = json.loads(repair_l_path.read_text())
            hero2_repair(
                repair_l,
                out_dir,
                name=f"l{ly}_repair_scatter",
                title_note=f" (layer-{ly} re-read)",
            )
        else:
            logger.warning("no %s — skipping l%d repair scatter", repair_l_path, ly)
    logger.info("figures complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
