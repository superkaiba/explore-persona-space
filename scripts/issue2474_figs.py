"""Issue #2474 P-C — figure driver over the P-B prefit outputs (plan v5 section 6).

Reads ``prefit_stats.json`` + ``prefit_scores.json`` (written by
``scripts/issue2474_fit.py`` phases scores/stats — the schema of record) and the
DV rates (via the fit driver's own ``_load_rates``), and renders the hero +
exploratory figure dump to ``figures/issue_2474/prefit_*.png`` (paper-plots
conventions; ``savefig_paper`` writes the ``.meta.json`` sidecar per figure,
augmented here with error-bar definitions + input provenance).

Figure registry (``--only`` takes comma-separated slugs):
  hero            grouped pooled-rho bars per setting at the pinned layer (arms +
                  competitors + the two real-answer ceilings), bootstrap CIs,
                  per-condition dots (the per-unit companion at condition grain)
  layers          pooled rho-by-layer curves per arm family, pinned layer marked
  scatter_ctx     per-trigger scatter (context arm vs level DV) per condition,
                  points labeled by trigger, inoculation prompt marked
  scatter_trainref  same for the Train-Ref predicted-answer arm
  pinoc_paired    all-triggers vs leave-inoculation-prompt-out paired bars
  dv_companion    level-DV vs change-DV paired bars
  centered        centered-variant rho-by-layer curves beside the base arms
  postft_forest   per-condition forest of delta-rho(base − post-ft) with CIs
  perm_band       observed max-over-layers vs the joint-permutation null band
  identbias_paired  fitted-map arms vs their shift-only (identity+bias) controls

Conventions (binding): one color = one meaning across every figure (module-level
arm->color map); NO fig.text caption/provenance blocks on the canvas; error-bar
definitions live in the sidecar ``figure_notes``, not on the canvas; xerr
offsets are clamped non-negative (repo errorbar convention).

Smoke (``--smoke``): renders from the fit driver's OWN synthetic smoke tree
(``issue2474_fit.py --phase smoke`` outputs under ``--smoke-dir``) into a /tmp
out-dir ONLY — zero writes under figures/ or eval_results/. Fails loud naming
the generation command when the tree is absent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from math import ceil
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "src" / "explore_persona_space"
    if not sentinel.is_dir():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} not found (parents[1]={root})")
    for p in (str(root), str(root / "src"), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

# Shared-VM thread caps (#847): load_dotenv() must precede every heavy import
# (tests/test_shared_vm_thread_caps.py), and the Agg pin must precede pyplot.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import issue2474_fit as fit_mod  # noqa: E402  (schema-of-record producer module)
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue2474_figs")

GEOMETRY_FAMS = tuple(fit_mod.GEOMETRY_FAMS)
TEXT_FAMS = tuple(fit_mod.TEXT_FAMS)

# Plain-English arm names (plan section 5 table) — canvas labels, never slugs.
ARM_NAMES: dict[str, str] = {
    "ctx_sameq": "Context state vs inoc. prompt",
    "ans_sameq_mapB": "Predicted answer vs inoc. prompt",
    "identbias_sameq": "Shift-only answer vs inoc. prompt",
    "ceiling_sameq": "Real answer vs inoc. prompt (ceiling)",
    "ctx_trainref": "Context state vs training contexts",
    "ans_trainref_mapB": "Predicted answer vs training answers",
    "identbias_trainref": "Shift-only answer vs training answers",
    "ceiling_trainref": "Real answer vs training answers (ceiling)",
    "propensity": "Base behavior rate",
    "bge_cos": "Text similarity (BGE)",
    "jaccard": "Text similarity (Jaccard)",
    "seqmatcher": "Text similarity (SequenceMatcher)",
    "tfidf_cos": "Text similarity (TF-IDF)",
}
SHORT_NAMES: dict[str, str] = {
    "ctx_sameq": "Context vs inoc.",
    "ans_sameq_mapB": "Pred. answer vs inoc.",
    "identbias_sameq": "Shift-only vs inoc.",
    "ceiling_sameq": "Real answer vs inoc.",
    "ctx_trainref": "Context vs train ctx.",
    "ans_trainref_mapB": "Pred. answer vs train ans.",
    "identbias_trainref": "Shift-only vs train ans.",
    "ceiling_trainref": "Real answer vs train ans.",
    "propensity": "Base behavior rate",
    "bge_cos": "Text sim. (BGE)",
}
SETTING_NAMES = {"em": "Emergent misalignment", "caps": "Capitalization"}
# Scatter label deconfliction: triggers below this re-elicitation rate form the
# per-panel "floor cluster" — count-annotated, not individually labeled.
FLOOR_LABEL_THR = 0.05
COND_NAMES = {
    "em_bad_medical_advice": "Bad medical advice",
    "em_bad_legal_advice": "Bad legal advice",
    "em_bad_security_advice": "Bad security advice",
    "em_turner_extreme_sports": "Extreme-sports advice",
    "em_turner_risky_financial": "Risky-financial advice",
    "caps_french": "French (caps)",
    "caps_german": "German (caps)",
    "caps_spanish": "Spanish (caps)",
}

# One color = one meaning, fixed across every figure in this driver: the 8 curated
# palette colors go to the geometry arms; competitors get fixed non-palette hexes
# (text baselines = gray-blue family, one shade per baseline; propensity = brown).
_PALETTE = paper_palette(8)
ARM_COLORS: dict[str, str] = {
    **{key: _PALETTE[i] for i, key in enumerate(GEOMETRY_FAMS)},
    "propensity": "#6d4c41",
    "bge_cos": "#455a64",
    "jaccard": "#78909c",
    "seqmatcher": "#90a4ae",
    "tfidf_cos": "#b0bec5",
}
NEUTRAL = "0.45"
HERO_ROWS = [*GEOMETRY_FAMS, "propensity", "bge_cos"]

CI_DEF = (
    "95% bootstrap CI (percentile over valid draws; one shared trigger-index multiset per "
    "setting x variant, seeds in prefit_stats.json .seeds; xerr rendered as non-negative "
    "offsets clamped at 0 per repo errorbar convention)"
)
P_INOC_CAVEAT = (
    "sameq-family predictors are identically 1 at the inoculation-prompt trigger by "
    "self-similarity; the leave-p_inoc-out variant is the registered sensitivity"
)


def _setting_name(s: str) -> str:
    return SETTING_NAMES.get(s, s)


def _cond_name(c: str) -> str:
    return COND_NAMES.get(c, c.replace("_", " "))


def _err_from_ci(points: list[float], cis: list[list[float]]) -> np.ndarray:
    """(2, N) non-negative xerr/yerr offsets from [lo, hi] CI pairs (clamped at 0)."""
    v = np.asarray(points, dtype=float)
    lo = np.asarray([c[0] for c in cis], dtype=float)
    hi = np.asarray([c[1] for c in cis], dtype=float)
    return np.vstack([np.maximum(0.0, v - lo), np.maximum(0.0, hi - v)])


def _ceiling_ref(stats: dict, setting: str, variant: str) -> float | None:
    """Banked round-1 DV cross-condition agreement ceiling (recomputed+asserted by stats)."""
    key = f"{setting}/{'with_p_inoc' if variant == 'full' else 'without_p_inoc'}"
    rec = stats.get("round1_recompute", {}).get(key)
    return float(rec["ceiling_mean"]) if rec else None


# ---------------------------------------------------------------------------
# Figure builders — each yields (stem, fig, notes)
# ---------------------------------------------------------------------------
def build_hero(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    stats = ctx["stats"]
    settings = list(stats["settings"])
    fig, axes = plt.subplots(
        1, len(settings), figsize=(6.4 * len(settings), 5.6), constrained_layout=True
    )
    axes_list = list(np.atleast_1d(axes))
    for ax, s in zip(axes_list, settings):
        sb = stats["settings"][s]
        vb = sb["variants"]["full"]
        conds = sb["conditions"]
        names, points, cis, colors, dots = [], [], [], [], []
        for key in HERO_ROWS:
            if key in vb["families"]:
                fe = vb["families"][key]
                p = fe["pooled"]["level"]["pinned"]
                points.append(float(p["rho"]))
                cis.append(p["ci95"])
                dots.append([fe["per_condition"][c]["level"]["pinned_rho"] for c in conds])
            elif key in vb.get("competitors", {}):
                ce = vb["competitors"][key]
                pc = [ce["per_condition"][c]["level_rho"] for c in conds]
                points.append(float(np.nanmean(pc)))
                cis.append(ce["pooled_level_ci95"])
                dots.append(pc)
            else:
                continue
            names.append(ARM_NAMES[key])
            colors.append(ARM_COLORS[key])
        y = np.arange(len(names))[::-1]
        ax.barh(
            y,
            points,
            xerr=_err_from_ci(points, cis),
            color=colors,
            height=0.62,
            error_kw={"elinewidth": 1.1, "capsize": 2.5, "ecolor": "0.25"},
        )
        for yi, dd in zip(y, dots):
            ax.plot(dd, np.full(len(dd), float(yi)), "o", ms=3.5, color="0.2", alpha=0.75, zorder=5)
        ceiling = _ceiling_ref(stats, s, "full")
        if ceiling is not None:
            ax.axvline(ceiling, ls="--", lw=1.2, color=NEUTRAL)
        ax.axvline(0.0, lw=0.8, color="0.6")
        ax.set_yticks(y, names)
        ax.set_xlabel("Pooled Spearman rho")
        ax.set_title(f"{_setting_name(s)} — layer {sb['pinned_layer']}", loc="left")
    handles = [
        Line2D([], [], marker="o", ls="", color="0.2", ms=4, label="Per-condition rho"),
        Line2D([], [], ls="--", color=NEUTRAL, label="DV agreement ceiling (banked)"),
    ]
    axes_list[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    notes = {
        "variant": "full",
        "dv": "level",
        "errorbar_definition": CI_DEF,
        "per_unit_companion": "per-condition dots on each pooled bar; per-trigger companions "
        "in the prefit_scatter_* figures",
        "p_inoc_caveat": P_INOC_CAVEAT,
    }
    return [("prefit_hero_pooled", fig, notes)]


def build_layers(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    out = []
    for s, sb in ctx["stats"]["settings"].items():
        vb = sb["variants"]["full"]
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for fam in GEOMETRY_FAMS:
            curve = vb["families"][fam]["pooled"]["level"]["rho_by_layer"]
            ax.plot(range(len(curve)), curve, color=ARM_COLORS[fam], lw=1.6, label=ARM_NAMES[fam])
        ax.axvline(sb["pinned_layer"], ls=":", lw=1.0, color=NEUTRAL)
        ax.axhline(0.0, lw=0.8, color="0.6")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Pooled Spearman rho (level DV)")
        ax.set_title(f"{_setting_name(s)} — rho by layer (all triggers)", loc="left")
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
        fig.subplots_adjust(right=0.66)
        notes = {
            "variant": "full",
            "dv": "level",
            "errorbar_definition": "none rendered; a 95% bootstrap confidence interval per "
            "layer for every family is stored in the committed stats JSON",
            "selection_caveat": "exploratory sweep; any max-over-layer quote carries the "
            "permutation band (prefit_perm_band_*), per plan section 6",
            "pinned_layer_marker": sb["pinned_layer"],
        }
        out.append((f"prefit_layers_{s}", fig, notes))
    return out


def _build_scatter(ctx: dict, fam: str, slug: str) -> list[tuple[str, "plt.Figure", dict]]:
    out = []
    for s, sb in ctx["stats"]["settings"].items():
        conds = sb["conditions"]
        pin = sb["pinned_layer"]
        vb = sb["variants"]["full"]
        ncols = min(3, len(conds))
        nrows = ceil(len(conds) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), squeeze=False, constrained_layout=True
        )
        for i, c in enumerate(conds):
            ax = axes.flat[i]
            sc = ctx["scores"]["conditions"][c]
            labels = sc["trigger_labels"]
            p_idx = int(sc["p_inoc_trigger_idx"])
            xs = np.array(
                [np.nan if v is None else float(v) for v in sc["families_layered"][fam][pin]]
            )
            ys = np.array([float(ctx["rates_level"][s][c][lab]) for lab in labels])
            ax.scatter(xs, ys, s=22, color=ARM_COLORS[fam], zorder=3)
            ax.scatter(
                xs[p_idx], ys[p_idx], s=90, marker="*", color="0.15", zorder=4
            )  # inoculation prompt
            # Label deconfliction (reconciler concern
            # scatter-label-deconfliction-before-body-embed): label only the
            # inoculation-prompt anchor + non-floor triggers; the floor cluster
            # gets ONE count note per panel (on the axis label — collision-free).
            # Cutoff = max(absolute FLOOR_LABEL_THR, bottom 15% of the panel's
            # rate range) so near-floor piles in panels with no exact-zero rows
            # (the EM conditions) deconflict too; per-point data stays in the
            # .meta.json sidecar.
            finite_ys = ys[np.isfinite(xs)]
            cutoff = max(
                FLOOR_LABEL_THR,
                float(finite_ys.min()) + 0.15 * float(finite_ys.max() - finite_ys.min()),
            )
            n_floor = 0
            for j, lab in enumerate(labels):
                if not np.isfinite(xs[j]):
                    continue
                if j != p_idx and ys[j] < cutoff:
                    n_floor += 1
                    continue
                short = lab if len(lab) <= 16 else lab[:15] + "…"
                ax.annotate(
                    short,
                    (xs[j], ys[j]),
                    fontsize=5.5,
                    alpha=0.8,
                    xytext=(2, 2),
                    textcoords="offset points",
                )
            rho = vb["families"][fam]["per_condition"][c]["level"]["pinned_rho"]
            ax.set_title(f"{_cond_name(c)} (rho={rho:.2f})", loc="left", fontsize=9)
            xlab = ARM_NAMES[fam]
            if n_floor:
                # Count note rides the axis label — collision-free with points/labels.
                xlab += f"\n({n_floor} near-floor triggers at rate < {cutoff:.2f} unlabeled)"
            ax.set_xlabel(xlab, fontsize=8)
            ax.set_ylabel("Per-trigger rate", fontsize=8)
        for k in range(len(conds), nrows * ncols):
            axes.flat[k].set_visible(False)
        notes = {
            "arm": ARM_NAMES[fam],
            "layer": pin,
            "dv": "level",
            "errorbar_definition": "none (per-trigger scatter; the per-unit companion to the "
            "pooled hero bars)",
            "marker_note": "star = the inoculation-prompt trigger",
            "label_policy": "anchor + non-floor triggers labeled; near-floor cluster "
            f"(rate < max({FLOOR_LABEL_THR:g}, bottom 15% of panel rate range)) "
            "count-noted on the axis label (all points plotted; per-point data in "
            "this sidecar)",
            "p_inoc_caveat": P_INOC_CAVEAT,
        }
        out.append((f"prefit_{slug}_{s}", fig, notes))
    return out


def build_scatter_ctx(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    return _build_scatter(ctx, "ctx_sameq", "scatter_ctx")


def build_scatter_trainref(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    return _build_scatter(ctx, "ans_trainref_mapB", "scatter_trainref")


def _grouped_bars(
    stats: dict,
    setting: str,
    picks: list[tuple[str, str]],
    bar_labels: tuple[str, str],
    second_style: dict,
) -> tuple["plt.Figure", "plt.Axes"]:
    """Two-bars-per-family grouped chart over (variant, dv) picks at the pinned layer."""
    sb = stats["settings"][setting]
    x = np.arange(len(GEOMETRY_FAMS))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for k, (variant, dv) in enumerate(picks):
        vb = sb["variants"][variant]
        points, cis = [], []
        for fam in GEOMETRY_FAMS:
            p = vb["families"][fam]["pooled"][dv]["pinned"]
            points.append(float(p["rho"]))
            cis.append(p["ci95"])
        kw = dict(second_style) if k == 1 else {}
        colors = [ARM_COLORS[f] for f in GEOMETRY_FAMS]
        ax.bar(
            x + (k - 0.5) * 0.38,
            points,
            width=0.36,
            color=colors,
            yerr=_err_from_ci(points, cis),
            error_kw={"elinewidth": 1.0, "capsize": 2.0, "ecolor": "0.25"},
            label=bar_labels[k],
            **kw,
        )
    ax.axhline(0.0, lw=0.8, color="0.6")
    ax.set_xticks(x, [SHORT_NAMES[f] for f in GEOMETRY_FAMS], rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Pooled Spearman rho")
    ax.legend(frameon=False, fontsize=8)
    return fig, ax


def build_pinoc_paired(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    out = []
    for s in ctx["stats"]["settings"]:
        fig, ax = _grouped_bars(
            ctx["stats"],
            s,
            [("full", "level"), ("loo", "level")],
            ("All triggers", "Leave inoculation prompt out"),
            {"alpha": 0.5},
        )
        ax.set_title(
            f"{_setting_name(s)} — sensitivity to the inoculation-prompt trigger", loc="left"
        )
        notes = {
            "dv": "level",
            "errorbar_definition": CI_DEF,
            "style_note": "lighter bars = leave-p_inoc-out variant",
            "p_inoc_caveat": P_INOC_CAVEAT,
        }
        out.append((f"prefit_pinoc_paired_{s}", fig, notes))
    return out


def build_dv_companion(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    out = []
    for s in ctx["stats"]["settings"]:
        fig, ax = _grouped_bars(
            ctx["stats"],
            s,
            [("full", "level"), ("full", "change")],
            ("Rate DV (level)", "Change DV (post minus base)"),
            {"hatch": "///", "alpha": 0.75},
        )
        ax.set_title(f"{_setting_name(s)} — level vs change DV", loc="left")
        notes = {
            "variant": "full",
            "errorbar_definition": CI_DEF,
            "change_dv_caveat": "base propensity enters the change DV by construction "
            "(mechanical -1 coefficient); the change read is exploratory (plan section 6)",
        }
        out.append((f"prefit_dv_companion_{s}", fig, notes))
    return out


def build_centered(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    out = []
    groups = (GEOMETRY_FAMS[:4], GEOMETRY_FAMS[4:])
    for s, sb in ctx["stats"]["settings"].items():
        vb = sb["variants"]["full"]
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True, constrained_layout=True)
        for ax, fams in zip(axes, groups):
            for fam in fams:
                base = vb["families"][fam]["pooled"]["level"]["rho_by_layer"]
                cent = vb["families"][f"{fam}_centered"]["pooled"]["level"]["rho_by_layer"]
                ax.plot(
                    range(len(base)), base, color=ARM_COLORS[fam], lw=1.5, label=SHORT_NAMES[fam]
                )
                ax.plot(range(len(cent)), cent, color=ARM_COLORS[fam], lw=1.2, ls="--")
            ax.axvline(sb["pinned_layer"], ls=":", lw=1.0, color=NEUTRAL)
            ax.axhline(0.0, lw=0.8, color="0.6")
            ax.set_xlabel("Layer")
            ax.legend(fontsize=7, frameon=False)
        axes[0].set_ylabel("Pooled Spearman rho (level DV)")
        axes[0].set_title(f"{_setting_name(s)} — inoc.-prompt reference arms", loc="left")
        axes[1].set_title("Training-reference arms", loc="left")
        notes = {
            "variant": "full",
            "dv": "level",
            "style_note": "solid = base arm, dashed = centered variant (trigger-panel mean "
            "subtracted before cosine, the parent convention)",
            "errorbar_definition": "none rendered; CIs in prefit_stats.json",
        }
        out.append((f"prefit_centered_{s}", fig, notes))
    return out


def build_postft_forest(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    out = []
    for s, sb in ctx["stats"]["settings"].items():
        vb = sb["variants"]["full"]
        conds = sb["conditions"]
        rows: list[tuple[str, float, list[float], str]] = []
        for fam in GEOMETRY_FAMS:
            entry = vb["paired"].get(fam, {}).get("vs_postft")
            if not entry:
                continue
            for c in conds:
                rec = entry["per_condition"][c]
                point = float(rec["rho_base"]) - float(rec["rho_postft"])
                rows.append(
                    (
                        f"{SHORT_NAMES[fam]} — {_cond_name(c)}",
                        point,
                        rec["delta_ci95"],
                        ARM_COLORS[fam],
                    )
                )
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 0.3 * len(rows) + 1.8))
        y = np.arange(len(rows))[::-1]
        points = [r[1] for r in rows]
        err = _err_from_ci(points, [r[2] for r in rows])
        for yi, (_, pt, _, col), e_lo, e_hi in zip(y, rows, err[0], err[1]):
            ax.errorbar(
                pt,
                yi,
                xerr=[[e_lo], [e_hi]],
                fmt="o",
                ms=4,
                color=col,
                elinewidth=1.1,
                capsize=2.0,
            )
        ax.axvline(0.0, lw=0.9, color="0.4")
        ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
        ax.set_xlabel("delta rho (base minus post-fine-tuning), level DV")
        ax.set_title(f"{_setting_name(s)} — base-model vs post-ft predictors", loc="left")
        fig.subplots_adjust(left=0.42)
        notes = {
            "variant": "full",
            "dv": "level",
            "errorbar_definition": CI_DEF + "; paired per-draw differences (shared trigger "
            "resample) vs the parent's committed post-ft predictor values",
            "point_definition": "rho_base - rho_postft at the pinned layer",
        }
        out.append((f"prefit_postft_forest_{s}", fig, notes))
    return out


def build_perm_band(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    out = []
    for s, sb in ctx["stats"]["settings"].items():
        vb = sb["variants"]["full"]
        fams = [f for f in GEOMETRY_FAMS if f in vb.get("permutation", {})]
        if not fams:
            continue
        fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(fams) + 1.6))
        y = np.arange(len(fams))[::-1]
        for yi, fam in zip(y, fams):
            b = vb["permutation"][fam]
            ax.plot(
                [b["null_max_p50"], b["null_max_p975"]],
                [yi, yi],
                color="0.75",
                lw=7,
                solid_capstyle="butt",
                zorder=2,
            )
            ax.plot(b["null_max_p95"], yi, "|", color="0.35", ms=13, zorder=3)
            obs = b["observed_pooled_max_over_layers"]
            ci = b.get("observed_max_ci95_selection_inherited")
            if obs is not None and ci is not None and None not in ci:
                ax.errorbar(
                    obs,
                    yi,
                    xerr=_err_from_ci([obs], [ci]),
                    fmt="none",
                    ecolor=ARM_COLORS[fam],
                    elinewidth=1.4,
                    capsize=2.5,
                    zorder=4,
                )
            if obs is not None:
                ax.plot(obs, yi, "o", color=ARM_COLORS[fam], ms=6, zorder=5)
        ceiling = _ceiling_ref(ctx["stats"], s, "full")
        if ceiling is not None:
            ax.axvline(ceiling, ls="--", lw=1.2, color=NEUTRAL)
        ax.set_yticks(y, [SHORT_NAMES[f] for f in fams], fontsize=8)
        ax.set_xlabel("Pooled Spearman rho (max over layers, level DV)")
        ax.set_title(f"{_setting_name(s)} — observed max vs permutation null band", loc="left")
        handles = [
            Line2D([], [], color="0.75", lw=7, label="Null max band (p50–p97.5)"),
            Line2D([], [], marker="|", ls="", color="0.35", ms=12, label="Null p95"),
            Line2D([], [], marker="o", ls="", color="0.2", label="Observed max ± 95% boot CI"),
            Line2D([], [], ls="--", color=NEUTRAL, label="DV agreement ceiling"),
        ]
        ax.legend(handles=handles, fontsize=7, frameon=False, loc="best")
        notes = {
            "variant": "full",
            "dv": "level",
            "band_definition": "per-draw max over layers of pooled signed rho under joint "
            "trigger permutation (one permutation per draw shared across arms/layers/"
            "conditions; n_perm in prefit_stats.json .seeds) — the selection-symmetric "
            "read for any max-over-layer quote",
            "errorbar_definition": "grey band = permutation-null percentiles (p50–p97.5), "
            "not a CI; colored bar on the observed point = selection-inherited 95% "
            "bootstrap CI (2.5/97.5 pct of per-draw max-over-layers of the "
            "condition-pooled rho, valid draws only)",
        }
        out.append((f"prefit_perm_band_{s}", fig, notes))
    return out


def build_identbias_paired(ctx: dict) -> list[tuple[str, "plt.Figure", dict]]:
    pairs = [("ans_sameq_mapB", "identbias_sameq"), ("ans_trainref_mapB", "identbias_trainref")]
    out = []
    for s, sb in ctx["stats"]["settings"].items():
        vb = sb["variants"]["full"]
        conds = sb["conditions"]
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        xticks, xticklabels = [], []
        pos = 0.0
        for map_fam, ib_fam in pairs:
            for k, fam in enumerate((map_fam, ib_fam)):
                p = vb["families"][fam]["pooled"]["level"]["pinned"]
                ax.bar(
                    pos,
                    float(p["rho"]),
                    width=0.8,
                    color=ARM_COLORS[fam],
                    yerr=_err_from_ci([float(p["rho"])], [p["ci95"]]),
                    error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "0.25"},
                )
                dots = [
                    vb["families"][fam]["per_condition"][c]["level"]["pinned_rho"] for c in conds
                ]
                ax.plot(
                    np.full(len(dots), pos), dots, "o", ms=3.5, color="0.2", alpha=0.75, zorder=5
                )
                xticks.append(pos)
                xticklabels.append(SHORT_NAMES[fam])
                pos += 1.0
            pos += 0.7
        ax.axhline(0.0, lw=0.8, color="0.6")
        ax.set_xticks(xticks, xticklabels, rotation=22, ha="right", fontsize=8)
        ax.set_ylabel("Pooled Spearman rho (level DV)")
        ax.set_title(f"{_setting_name(s)} — fitted map vs identity+bias control", loc="left")
        notes = {
            "variant": "full",
            "dv": "level",
            "errorbar_definition": CI_DEF,
            "control_definition": "shift-only = identity+learned-bias map applied in place of "
            "the fitted base map (mapping-baselines control, plan section 5)",
        }
        out.append((f"prefit_identbias_{s}", fig, notes))
    return out


FIGS: dict[str, object] = {
    "hero": build_hero,
    "layers": build_layers,
    "scatter_ctx": build_scatter_ctx,
    "scatter_trainref": build_scatter_trainref,
    "pinoc_paired": build_pinoc_paired,
    "dv_companion": build_dv_companion,
    "centered": build_centered,
    "postft_forest": build_postft_forest,
    "perm_band": build_perm_band,
    "identbias_paired": build_identbias_paired,
}


# ---------------------------------------------------------------------------
# IO + dispatch
# ---------------------------------------------------------------------------
def _augment_sidecar(meta_path: Path, notes: dict) -> None:
    """Add figure_notes to savefig_paper's sidecar (render_id etc. untouched)."""
    payload = json.loads(meta_path.read_text())
    payload["figure_notes"] = notes
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def _resolve_inputs(args) -> tuple[Path, Path, dict]:
    """Resolve (stats_path, scores_path, rates) for smoke vs production."""
    smoke_root = Path(args.smoke_dir)
    if args.stats:
        stats_path = Path(args.stats)
    elif args.smoke:
        stats_path = smoke_root / "out" / "prefit_stats.json"
    else:
        stats_path = REPO_ROOT / "eval_results" / "issue_2474" / "prefit" / "prefit_stats.json"
    scores_path = Path(args.scores) if args.scores else stats_path.with_name("prefit_scores.json")
    for p, phase_hint in ((stats_path, "stats"), (scores_path, "scores")):
        if not p.is_file():
            hint = (
                "generate the smoke fixtures first: uv run python scripts/issue2474_fit.py "
                "--phase smoke"
                if args.smoke
                else f"produced by issue2474_fit.py --phase {phase_hint} (plan section 4 P-B)"
            )
            raise RuntimeError(f"input missing: {p} — {hint}")
    if args.rates:
        rates_path: Path | None = Path(args.rates)
    elif args.smoke:
        rates_path = smoke_root / "rates_synth.json"
    else:
        rates_path = None  # production: the fit driver's pinned free_gate loaders
    rates_level = fit_mod._load_rates({"rates_path": rates_path}, "level")
    return stats_path, scores_path, rates_level


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats", default=None, help="prefit_stats.json (plan section 10 command)")
    ap.add_argument("--scores", default=None, help="prefit_scores.json (default: sibling)")
    ap.add_argument(
        "--rates",
        default=None,
        help="explicit {level,cont} rates JSON (default: pinned parent artifacts; "
        "smoke: <smoke-dir>/rates_synth.json)",
    )
    ap.add_argument("--out-dir", default=None, help="default figures/issue_2474 (smoke: /tmp)")
    ap.add_argument(
        "--smoke", action="store_true", help="render from the fit smoke tree, /tmp only"
    )
    ap.add_argument("--smoke-dir", default="/tmp/issue2474_smoke")
    ap.add_argument("--only", default=None, help=f"comma-separated slugs from {sorted(FIGS)}")
    ap.add_argument("--list-figs", action="store_true", help="print the registry and exit")
    ap.add_argument(
        "--import-check", action="store_true", help="argcheck + call-arity bind, then exit 0"
    )
    return ap


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.list_figs:
        for slug in FIGS:
            print(slug)
        return 0

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            Path("/tmp/issue2474_figs_smoke")
            if args.smoke
            else REPO_ROOT / "figures" / "issue_2474"
        )
    )
    if args.smoke and not str(out_dir).startswith("/tmp/"):
        raise RuntimeError(
            f"--smoke must render into /tmp only (got --out-dir {out_dir}); zero writes "
            "under figures/ or eval_results/ in smoke mode"
        )

    stats_path, scores_path, rates_level = _resolve_inputs(args)
    stats = json.loads(stats_path.read_text())
    scores = json.loads(scores_path.read_text())
    ctx = {"stats": stats, "scores": scores, "rates_level": rates_level}

    selected = list(FIGS) if not args.only else [s.strip() for s in args.only.split(",")]
    unknown = [s for s in selected if s not in FIGS]
    if unknown:
        raise SystemExit(f"unknown --only slug(s) {unknown}; available: {sorted(FIGS)}")

    set_paper_style("blog")
    common_notes = {
        "issue": 2474,
        "inputs": {
            "stats": str(stats_path),
            "scores": str(scores_path),
            "rates": str(args.rates)
            if args.rates
            else ("smoke:rates_synth.json" if args.smoke else "pinned parent rate artifacts"),
        },
        "stats_git_commit": stats.get("git", {}).get("git_commit"),
        "parent_sha": stats.get("parent_sha"),
        "seeds": stats.get("seeds"),
    }
    n_written = 0
    for slug in selected:
        for stem, fig, notes in FIGS[slug](ctx):
            paths = savefig_paper(fig, stem, dir=out_dir)
            plt.close(fig)
            _augment_sidecar(paths["meta"], {**common_notes, "figure": slug, **notes})
            png = paths.get("png")
            if png is None or not Path(png).is_file() or Path(png).stat().st_size == 0:
                raise RuntimeError(f"figure {stem}: PNG missing/empty after savefig_paper")
            n_written += 1
            print(f"[figs] wrote {png} ({Path(png).stat().st_size} bytes)", flush=True)
    if n_written == 0:
        raise RuntimeError("no figures rendered — empty selection over the registry")
    print(f"[figs] done — {n_written} figure(s) under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
