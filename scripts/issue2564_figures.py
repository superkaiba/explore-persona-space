"""Issue #2564 PE figures: hero axis-profile + exploratory dump.

Reads the PE analysis outputs (``minpair_delta.json`` + ``perpair.jsonl``,
written by ``scripts/issue2564_analysis.py``) and renders the plan §6 figure
set under ``figures/issue_2564/`` via the paper-plots conventions
(``set_paper_style("blog")`` + ``savefig_paper`` -> PNG + PDF + meta.json).

Hero: ``fig_hero_axis_profile`` — one row per axis, four aligned panels
(direction cosine with ceiling/null/identity whiskers; calibration ratio to
the global slope; axis-identity cosine; text-vs-representation flip norm,
paraphrase-normalized). Exploratory figures each land in their own file and
skip gracefully (one logged line) when their input read is n/a — e.g. layer
twins exist for the identity baseline only, and cross-family reads are null
on single-family axes.

Smoke mode (``--smoke``) consumes a smoke out-root and writes to /tmp —
never the committed ``figures/`` tree.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before matplotlib/numpy: thread caps + headless env on the shared VM

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue2564 import paths as P2564  # noqa: E402

# Shared with scripts/issue2564_analysis.py via experiments.issue2564.paths
# (r2 blocker 4): producer default out-dir == consumer default results-dir,
# repo-root-anchored — never cwd-relative, in BOTH modes.
SMOKE_ROOT = P2564.SMOKE_ROOT
DEFAULT_RESULTS_DIR = P2564.production_results_dir()
SMOKE_RESULTS_DIR = P2564.smoke_results_dir()
DEFAULT_OUT_DIR = P2564.repo_root() / "figures"
STEM_PREFIX = "issue_2564"
_SINGLE_CARRIER_RE = re.compile(r"^c\d+$")  # excludes query_content dyads ("c01|c02")

ARMS = ("arm_779ce", "arm_1738ce", "arm_iddelta")
# Plain-English labels only on rendered text (paper-plots SKILL §3.5).
ARM_LABELS = {
    "arm_779ce": "single-turn frozen map",
    "arm_1738ce": "multi-turn frozen map",
    "arm_iddelta": "identity baseline",
}
PAIR_CLASS_LABELS = {
    "install": "install",
    "swap": "value swap",
    "famswap": "family swap",
    "instruction_paraphrase": "instruction paraphrase",
    "query_content": "query content",
    "query_form": "query form",
    "query_paraphrase": "query paraphrase",
}


def axis_label(axis: str) -> str:
    """Plain-English axis name for rendered figure text."""
    return axis.replace("_", " ")


def _get(d: object, *keys: str) -> object:
    """Nested dict lookup returning None on any missing/None hop."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return None
        cur = cur[k]
    return cur


def _finite(x: object) -> float | None:
    """Return float(x) when finite, else None."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _xerr(v: float, ci: object) -> tuple[list[float], list[float]]:
    """Non-negative errorbar offsets from a [lo, hi] CI (clamped; gotchas.md)."""
    if not isinstance(ci, list | tuple) or len(ci) != 2:
        return [0.0], [0.0]
    lo, hi = _finite(ci[0]), _finite(ci[1])
    if lo is None or hi is None:
        return [0.0], [0.0]
    return [max(0.0, v - lo)], [max(0.0, hi - v)]


def _arm_colors() -> dict[str, str]:
    return {
        "arm_779ce": paper_palette_role("primary"),
        "arm_1738ce": paper_palette_role("baseline"),
        "arm_iddelta": paper_palette_role("control"),
    }


def _class_colors() -> dict[str, str]:
    order = sorted(PAIR_CLASS_LABELS)
    cols = paper_palette_blog(len(order))
    return dict(zip(order, cols, strict=True))


def _axes_sorted(doc: dict) -> list[str]:
    """Instruction axes first (alphabetical), query axes last."""
    names = sorted(doc.get("axes", {}))
    return [a for a in names if not a.startswith("query")] + [
        a for a in names if a.startswith("query")
    ]


def _save(fig: plt.Figure, stem: str, out_dir: Path) -> str:
    savefig_paper(fig, f"{STEM_PREFIX}/{stem}", dir=out_dir)
    plt.close(fig)
    return stem


# ── hero ──────────────────────────────────────────────────────────────


def fig_hero_axis_profile(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Hero: per-axis profile — direction / calibration / identity / text-vs-repr."""
    axes_names = _axes_sorted(doc)
    if not axes_names:
        return None
    n = len(axes_names)
    colors = _arm_colors()
    neutral = paper_palette_role("neutral")
    fig, panels = plt.subplots(1, 4, figsize=(13.0, max(3.2, 0.75 * n + 1.6)), sharey=True)
    ys = np.arange(n)[::-1]  # first axis at top
    off = {"arm_779ce": 0.22, "arm_1738ce": 0.0, "arm_iddelta": -0.22}

    # Panel 1: direction cosine (headline pairs) + null band + split-half ceiling.
    ax = panels[0]
    for i, name in enumerate(axes_names):
        d = _get(doc, "axes", name, "direction")
        if not isinstance(d, dict):
            continue
        nul = _get(d, "arm_779ce", "null")
        if isinstance(nul, dict):
            q_lo, q_hi = _finite(nul.get("q2_5")), _finite(nul.get("q97_5"))
            if q_lo is not None and q_hi is not None:
                ax.plot([q_lo, q_hi], [ys[i], ys[i]], color=neutral, lw=5, alpha=0.35, zorder=1)
        for arm in ARMS:
            v = _finite(_get(d, arm, "mean_cos_headline"))
            if v is None:
                continue
            lo, hi = _xerr(v, _get(d, arm, "ci95"))
            ax.errorbar(
                [v],
                [ys[i] + off[arm]],
                xerr=[lo, hi],
                fmt="o",
                ms=5,
                color=colors[arm],
                ecolor=colors[arm],
                elinewidth=1.2,
                capsize=2,
                zorder=3,
            )
        # Split-half ceiling: DIRECT read of reliability.r10_mean + its CI
        # (r2 concern hero-ceiling-suppressed: the old reconstruction via
        # ceiling_normalized_cos vanished exactly on suppressed axes — the
        # rows where seeing the ceiling matters most).
        r10 = _finite(_get(doc, "axes", name, "reliability", "r10_mean"))
        if r10 is not None:
            lo_c, hi_c = _xerr(r10, _get(doc, "axes", name, "reliability", "r10_ci95"))
            ax.errorbar(
                [r10],
                [ys[i]],
                xerr=[lo_c, hi_c],
                marker="D",
                ms=6,
                mfc="none",
                mec=neutral,
                mew=1.4,
                ls="none",
                ecolor=neutral,
                elinewidth=1.0,
                capsize=2,
                zorder=2,
            )
    ax.axvline(0.0, color=neutral, lw=0.8, alpha=0.6)
    ax.set_xlabel("direction cosine")
    ax.set_title("direction recovery", loc="left")
    ax.set_yticks(np.arange(n)[::-1])
    ax.set_yticklabels([axis_label(a) for a in axes_names])

    # Panel 2: calibration ratio to the global slope.
    ax = panels[1]
    for i, name in enumerate(axes_names):
        c = _get(doc, "axes", name, "calibration")
        if not isinstance(c, dict):
            continue
        for arm in ARMS:
            v = _finite(_get(c, arm, "ratio_to_global"))
            if v is None:
                continue
            lo, hi = _xerr(v, _get(c, arm, "ratio_to_global_ci95"))
            ax.errorbar(
                [v],
                [ys[i] + off[arm]],
                xerr=[lo, hi],
                fmt="o",
                ms=5,
                color=colors[arm],
                ecolor=colors[arm],
                elinewidth=1.2,
                capsize=2,
            )
    ax.axvline(1.0, color=neutral, lw=0.8, alpha=0.6)
    ax.set_xlabel("slope ratio, axis / global")
    ax.set_title("calibration", loc="left")
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    # Panel 3: axis-identity cosine (median across value pairs).
    ax = panels[2]
    for i, name in enumerate(axes_names):
        idn = _get(doc, "axes", name, "identity")
        if not isinstance(idn, dict):
            continue
        for arm in ARMS:
            v = _finite(_get(idn, arm, "median"))
            if v is None:
                continue
            lo, hi = _xerr(v, _get(idn, arm, "median_ci95"))
            ax.errorbar(
                [v],
                [ys[i] + off[arm]],
                xerr=[lo, hi],
                fmt="o",
                ms=5,
                color=colors[arm],
                ecolor=colors[arm],
                elinewidth=1.2,
                capsize=2,
            )
    ax.axvline(0.0, color=neutral, lw=0.8, alpha=0.6)
    ax.set_xlabel("axis-identity cosine")
    ax.set_title("within-axis consistency", loc="left")

    # Panel 4: flip norm / paraphrase norm — text space vs representation space.
    ax = panels[3]
    text_c = paper_palette_role("accent")
    repr_c = "#444444"
    for i, name in enumerate(axes_names):
        t = _finite(_get(doc, "axes", name, "text_space", "flip_over_para_ratio"))
        if t is not None:
            ax.plot([t], [ys[i] + 0.15], marker="o", ms=5, color=text_c, ls="none")
        srf = _get(doc, "axes", name, "surface", "observed")
        fn = _finite(_get(srf, "flip_norm_mean")) if isinstance(srf, dict) else None
        pn = _finite(_get(srf, "para_norm_mean")) if isinstance(srf, dict) else None
        if fn is not None and pn is not None and pn > 0:
            ax.plot([fn / pn], [ys[i] - 0.15], marker="s", ms=5, color=repr_c, ls="none")
    ax.axvline(1.0, color=neutral, lw=0.8, alpha=0.6)
    ax.set_xlabel("flip / paraphrase norm")
    ax.set_title("text vs representation", loc="left")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.plot([], [], marker="o", color=text_c, ls="none", label="answer text (embedding)")
    ax.plot([], [], marker="s", color=repr_c, ls="none", label="representation (observed)")
    handles = [
        plt.Line2D([], [], marker="o", color=colors[a], ls="none", label=ARM_LABELS[a])
        for a in ARMS
    ]
    panels[0].legend(handles=handles, loc="lower left", fontsize=8)
    ax.legend(loc="lower right", fontsize=8)
    for p in panels:
        p.set_ylim(-0.6, n - 0.4)
    fig.suptitle("Per-axis profile of the frozen context-to-answer maps", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "fig_hero_axis_profile", out_dir)


# ── exploratory dump ──────────────────────────────────────────────────


def fig_norm_scatter(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Per-axis predicted-vs-observed shift-norm scatter with the global slope line."""
    axes_names = _axes_sorted(doc)
    pts = [r for r in rows if _finite(_get(r, "norm_pred", "arm_779ce")) is not None]
    if not axes_names or not pts:
        return None
    ncol = min(4, len(axes_names))
    nrow = math.ceil(len(axes_names) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    slope = _finite(
        _get(doc, "axes", axes_names[0], "calibration", "arm_779ce", "global_slope_all2778")
    )
    col = _arm_colors()["arm_779ce"]
    neutral = paper_palette_role("neutral")
    for j, name in enumerate(axes_names):
        ax = axs[j // ncol][j % ncol]
        sub = [r for r in pts if r.get("axis") == name]
        xs = np.array([r["norm_obs_tail_L19"] for r in sub], dtype=float)
        ys_ = np.array([r["norm_pred"]["arm_779ce"] for r in sub], dtype=float)
        # in_headline_70 is floor-gated upstream (r3 headline-pair-floor-mislabel):
        # fired pairs on a compliance-limited axis land in the OPEN marker set.
        filled = np.array([bool(r.get("in_headline_70")) for r in sub])
        if filled.any():
            ax.scatter(xs[filled], ys_[filled], s=16, color=col, alpha=0.75, label="headline pair")
        if (~filled).any():
            ax.scatter(
                xs[~filled],
                ys_[~filled],
                s=16,
                facecolors="none",
                edgecolors=col,
                linewidths=1.0,
                alpha=0.75,
                label="non-headline pair",
            )
        finite_xs = xs[np.isfinite(xs)]
        if slope is not None and finite_xs.size:
            xline = np.linspace(0, max(1e-9, float(finite_xs.max())), 32)
            ax.plot(xline, slope * xline, color=neutral, lw=1.0, label="global slope")
        ax.set_title(axis_label(name), loc="left")
        ax.set_xlabel("observed shift norm")
        ax.set_ylabel("predicted shift norm")
        if j == 0:
            ax.legend(fontsize=7)
    for j in range(len(axes_names), nrow * ncol):
        axs[j // ncol][j % ncol].set_visible(False)
    fig.suptitle(
        "Predicted vs observed answer-shift magnitude (single-turn frozen map)", x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fig_expl_norm_scatter", out_dir)


def fig_install_vs_swap(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Per-axis distributions of per-pair direction cosine, split by pair class."""
    axes_names = _axes_sorted(doc)
    groups: list[tuple[str, str, list[float]]] = []
    for name in axes_names:
        for cls in sorted({r["pair_class"] for r in rows if r.get("axis") == name}):
            vals = [
                v
                for r in rows
                if r.get("axis") == name
                and r.get("pair_class") == cls
                and (v := _finite(_get(r, "cos", "arm_779ce"))) is not None
            ]
            if vals:
                groups.append((name, cls, vals))
    if not groups:
        return None
    ccol = _class_colors()
    fig, ax = plt.subplots(figsize=(max(6.5, 0.85 * len(groups)), 4.0))
    positions = np.arange(len(groups))
    parts = ax.violinplot([g[2] for g in groups], positions=positions, showmedians=True, widths=0.8)
    for body, (_, cls, _) in zip(parts["bodies"], groups, strict=True):
        body.set_facecolor(ccol[cls])
        body.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{axis_label(a)}\n{PAIR_CLASS_LABELS.get(c, c)}" for a, c, _ in groups], fontsize=7
    )
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8, alpha=0.6)
    ax.set_ylabel("per-pair direction cosine")
    ax.set_title("Direction cosine by axis and pair class (single-turn frozen map)", loc="left")
    fig.tight_layout()
    return _save(fig, "fig_expl_install_vs_swap_violin", out_dir)


def _vp_matrix(per_vp: dict[str, float]) -> tuple[list[str], np.ndarray] | None:
    names: list[str] = []
    for key in per_vp:
        parts = key.split("-")
        if len(parts) != 2:
            return None
        for p in parts:
            if p not in names:
                names.append(p)
    names.sort()
    mat = np.full((len(names), len(names)), np.nan)
    for key, v in per_vp.items():
        a, b = key.split("-")
        i, j = names.index(a), names.index(b)
        fv = _finite(v)
        if fv is None:
            continue
        mat[i, j] = fv
        mat[j, i] = fv
    return names, mat


def fig_identity_heatmaps(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Per-axis heatmaps of axis-identity cosine over value pairs (primary arm)."""
    axes_names = [
        a
        for a in _axes_sorted(doc)
        if isinstance(_get(doc, "axes", a, "identity", "arm_779ce", "per_vp_cos"), dict)
    ]
    if not axes_names:
        return None
    ncol = min(4, len(axes_names))
    nrow = math.ceil(len(axes_names) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
    im = None
    for j, name in enumerate(axes_names):
        ax = axs[j // ncol][j % ncol]
        parsed = _vp_matrix(_get(doc, "axes", name, "identity", "arm_779ce", "per_vp_cos"))
        if parsed is None:
            ax.set_visible(False)
            continue
        names, mat = parsed
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels([f"value {n_}" for n_ in names], fontsize=6, rotation=45, ha="right")
        ax.set_yticklabels([f"value {n_}" for n_ in names], fontsize=6)
        ax.set_title(axis_label(name), loc="left")
    for j in range(len(axes_names), nrow * ncol):
        axs[j // ncol][j % ncol].set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.7, label="axis-identity cosine")
    fig.suptitle("Axis-identity cosine per value pair (single-turn frozen map)", x=0.01, ha="left")
    return _save(fig, "fig_expl_identity_heatmaps", out_dir)


def fig_cross_family_scatter(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Observed vs predicted cross-family (template vs paraphrase) consistency."""
    pts: list[tuple[str, float, float]] = []
    for name in _axes_sorted(doc):
        obs = _get(doc, "axes", name, "cross_family", "observed", "per_vp_cos")
        prd = _get(doc, "axes", name, "cross_family", "arm_779ce", "per_vp_cos")
        if not isinstance(obs, dict) or not isinstance(prd, dict):
            continue
        for vp, ov in obs.items():
            o, p = _finite(ov), _finite(prd.get(vp))
            if o is not None and p is not None:
                pts.append((name, o, p))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    axes_present = sorted({a for a, _, _ in pts})
    cols = dict(zip(axes_present, paper_palette_blog(max(1, len(axes_present))), strict=False))
    for name in axes_present:
        xs = [o for a, o, _ in pts if a == name]
        ys_ = [p for a, _, p in pts if a == name]
        ax.scatter(xs, ys_, s=22, color=cols[name], label=axis_label(name), alpha=0.85)
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, color=paper_palette_role("neutral"), lw=0.8, alpha=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("observed cross-family consistency (cosine)")
    ax.set_ylabel("predicted cross-family consistency (cosine)")
    ax.set_title("Cross-family consistency: observed vs predicted", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig_expl_cross_family_scatter", out_dir)


def fig_edit_dose(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Predicted shift norm vs changed-token count, colored by pair class."""
    pts = [
        r
        for r in rows
        if _finite(r.get("changed_tokens")) is not None
        and _finite(_get(r, "norm_pred", "arm_779ce")) is not None
    ]
    if not pts:
        return None
    ccol = _class_colors()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for cls in sorted({r["pair_class"] for r in pts}):
        sub = [r for r in pts if r["pair_class"] == cls]
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(sub))
        ax.scatter(
            np.array([r["changed_tokens"] for r in sub], dtype=float) + jitter,
            [r["norm_pred"]["arm_779ce"] for r in sub],
            s=16,
            color=ccol[cls],
            alpha=0.7,
            label=PAIR_CLASS_LABELS.get(cls, cls),
        )
    ax.set_xlabel("changed tokens between the pair's contexts")
    ax.set_ylabel("predicted shift norm")
    ax.set_title("Edit dose vs predicted shift magnitude", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig_expl_edit_dose", out_dir)


def fig_retrieval_curves(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Delta-retrieval accuracy-at-k curves per arm (global pool)."""
    glob = _get(doc, "retrieval", "global")
    if not isinstance(glob, dict):
        return None
    colors = _arm_colors()
    neutral = paper_palette_role("neutral")
    metrics = ("cosine", "euclidean")
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    drew = False
    for ax, metric in zip(axs, metrics, strict=True):
        for arm in ARMS:
            acc = _get(glob, arm, metric, "acc_at_k")
            if not isinstance(acc, dict) or not acc:
                continue
            ks = sorted(int(k) for k in acc)
            ax.plot(
                ks,
                [acc[str(k)] for k in ks],
                marker="o",
                ms=4,
                color=colors[arm],
                label=ARM_LABELS[arm],
            )
            drew = True
        chance = _get(glob, "arm_779ce", metric, "chance_at_k")
        if isinstance(chance, dict) and chance:
            ks = sorted(int(k) for k in chance)
            ax.plot(ks, [chance[str(k)] for k in ks], ls="--", color=neutral, label="chance")
        ax.set_xlabel("k nearest neighbors")
        ax.set_title(f"{metric} distance", loc="left")
    if not drew:
        plt.close(fig)
        return None
    axs[0].set_ylabel("retrieval accuracy at k")
    axs[0].legend(fontsize=7)
    fig.suptitle(
        "Delta retrieval: does the predicted shift find its observed pair?", x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig_expl_retrieval_acc", out_dir)


def fig_flip_vs_para_text(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Per-axis distributions of answer-text shift norm: flips vs paraphrases."""
    para_cls = {"instruction_paraphrase", "query_paraphrase"}
    groups: list[tuple[str, str, list[float]]] = []
    for name in _axes_sorted(doc):
        # r2 concern exploratory-figure-pair-filters: the flip bucket is the
        # axis's PRIMARY class only (doc metadata) — never every non-para
        # class (install/famswap rows are controls, not flips).
        prim_cls = _get(doc, "axes", name, "primary_class")
        flips = [
            v
            for r in rows
            if r.get("axis") == name
            and r.get("pair_class") == prim_cls
            and (v := _finite(r.get("norm_text"))) is not None
        ]
        paras = [
            v
            for r in rows
            if r.get("axis") == name
            and r.get("pair_class") in para_cls
            and (v := _finite(r.get("norm_text"))) is not None
        ]
        if flips:
            groups.append((name, "value flip", flips))
        if paras:
            groups.append((name, "paraphrase", paras))
    if not groups:
        return None
    fig, ax = plt.subplots(figsize=(max(6.5, 0.85 * len(groups)), 4.0))
    kind_col = {
        "value flip": paper_palette_role("accent"),
        "paraphrase": paper_palette_role("neutral"),
    }
    positions = np.arange(len(groups))
    bp = ax.boxplot(
        [g[2] for g in groups],
        positions=positions,
        widths=0.7,
        patch_artist=True,
        medianprops={"color": "#333333"},
    )
    for patch, (_, kind, _) in zip(bp["boxes"], groups, strict=True):
        patch.set_facecolor(kind_col[kind])
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{axis_label(a)}\n{k}" for a, k, _ in groups], fontsize=7)
    ax.set_ylabel("answer-text shift norm (embedding space)")
    ax.set_title("Text-space shift: value flips vs paraphrases", loc="left")
    fig.tight_layout()
    return _save(fig, "fig_expl_flip_vs_para_text", out_dir)


def fig_carrier_transfer(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Axis-by-carrier matrix of mean per-pair direction cosine (primary arm)."""
    axes_names = _axes_sorted(doc)
    # r2 concern exploratory-figure-pair-filters: single carriers only
    # (query_content dyads "c01|c02" are not carrier cells), and per axis
    # only its PRIMARY pair class (controls would dilute the cell means).
    carriers = sorted(
        {
            r["carrier"]
            for r in rows
            if r.get("carrier") and _SINGLE_CARRIER_RE.fullmatch(str(r["carrier"]))
        }
    )
    if not axes_names or not carriers:
        return None
    mat = np.full((len(axes_names), len(carriers)), np.nan)
    for i, name in enumerate(axes_names):
        prim_cls = _get(doc, "axes", name, "primary_class")
        for j, car in enumerate(carriers):
            vals = [
                v
                for r in rows
                if r.get("axis") == name
                and r.get("pair_class") == prim_cls
                and r.get("carrier") == car
                and (v := _finite(_get(r, "cos", "arm_779ce"))) is not None
            ]
            if vals:
                mat[i, j] = float(np.mean(vals))
    if np.isnan(mat).all():
        return None
    fig, ax = plt.subplots(
        figsize=(max(5.0, 0.55 * len(carriers) + 2.5), max(3.2, 0.5 * len(axes_names) + 1.5))
    )
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(carriers)))
    ax.set_xticklabels(
        [f"carrier {c.lstrip('c').lstrip('0') or '0'}" for c in carriers], fontsize=7
    )
    ax.set_yticks(range(len(axes_names)))
    ax.set_yticklabels([axis_label(a) for a in axes_names], fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean direction cosine")
    ax.set_title("Direction cosine by axis and carrier question", loc="left")
    # No tight_layout after fig.colorbar: matplotlib refuses the layout-engine
    # switch under the paper style (RuntimeError: Colorbar layout ... not compatible).
    return _save(fig, "fig_expl_carrier_transfer", out_dir)


def fig_arm_agreement(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Per-pair agreement between the single-turn and multi-turn frozen maps."""
    pts = [
        (a, b)
        for r in rows
        if (a := _finite(_get(r, "cos", "arm_779ce"))) is not None
        and (b := _finite(_get(r, "cos", "arm_1738ce"))) is not None
    ]
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.scatter(
        [p[0] for p in pts], [p[1] for p in pts], s=14, color=_arm_colors()["arm_779ce"], alpha=0.6
    )
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, color=paper_palette_role("neutral"), lw=0.8, alpha=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("direction cosine, single-turn frozen map")
    ax.set_ylabel("direction cosine, multi-turn frozen map")
    ax.set_title("Map agreement per pair", loc="left")
    fig.tight_layout()
    return _save(fig, "fig_expl_arm_agreement", out_dir)


def fig_ceiling_vs_cos(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Per-axis split-half reliability vs headline direction cosine."""
    pts: list[tuple[str, float, float]] = []
    for name in _axes_sorted(doc):
        # r10_mean is the ceiling the headline reads are normalized by
        # (Spearman-Brown r10), matching the hero panel's ceiling marker.
        r10 = _finite(_get(doc, "axes", name, "reliability", "r10_mean"))
        v = _finite(_get(doc, "axes", name, "direction", "arm_779ce", "mean_cos_headline"))
        if r10 is not None and v is not None:
            pts.append((name, r10, v))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    col = _arm_colors()["arm_779ce"]
    for name, x, y in pts:
        ax.scatter([x], [y], s=26, color=col)
        ax.text(x, y, f" {axis_label(name)}", fontsize=7, va="center")
    ax.set_xlabel("reliability ceiling of the observed shift (Spearman-Brown r10)")
    ax.set_ylabel("direction cosine (single-turn frozen map)")
    ax.set_title("Reliability ceiling vs direction recovery", loc="left")
    fig.tight_layout()
    return _save(fig, "fig_expl_ceiling_vs_cos", out_dir)


def fig_query_form_dissociation(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Query axes: text-space vs representation-space vs predicted flip/para ratios."""
    q_axes = [a for a in _axes_sorted(doc) if a.startswith("query")]
    bars: list[tuple[str, str, float]] = []
    for name in q_axes:
        t = _finite(_get(doc, "axes", name, "text_space", "flip_over_para_ratio"))
        if t is not None:
            bars.append((name, "answer text", t))
        srf_o = _get(doc, "axes", name, "surface", "observed")
        if isinstance(srf_o, dict):
            fn = _finite(srf_o.get("flip_norm_mean"))
            pn = _finite(srf_o.get("para_norm_mean"))
            if fn is not None and pn is not None and pn > 0:
                bars.append((name, "representation (observed)", fn / pn))
        srf_p = _get(doc, "axes", name, "surface", "arm_779ce")
        if isinstance(srf_p, dict):
            fn = _finite(srf_p.get("flip_norm_mean"))
            pn = _finite(srf_p.get("para_norm_mean"))
            if fn is not None and pn is not None and pn > 0:
                bars.append((name, "representation (predicted)", fn / pn))
    if not bars:
        return None
    kinds = ["answer text", "representation (observed)", "representation (predicted)"]
    kcol = {
        "answer text": paper_palette_role("accent"),
        "representation (observed)": "#444444",
        "representation (predicted)": _arm_colors()["arm_779ce"],
    }
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    width = 0.25
    for ki, kind in enumerate(kinds):
        xs, hs = [], []
        for ai, name in enumerate(q_axes):
            v = next((v for a, k, v in bars if a == name and k == kind), None)
            if v is not None:
                xs.append(ai + (ki - 1) * width)
                hs.append(v)
        if xs:
            ax.bar(xs, hs, width=width, color=kcol[kind], label=kind)
    ax.axhline(1.0, color=paper_palette_role("neutral"), lw=0.8, alpha=0.6)
    ax.set_xticks(range(len(q_axes)))
    ax.set_xticklabels([axis_label(a) for a in q_axes])
    ax.set_ylabel("flip norm / paraphrase norm")
    ax.set_title("Query axes: text vs representation dissociation", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig_expl_query_dissociation", out_dir)


def fig_layer_twins(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Layer sensitivity of the identity baseline's direction cosine (14/19/26)."""
    axes_names = _axes_sorted(doc)
    series: list[tuple[str, list[int], list[float]]] = []
    for name in axes_names:
        pairs: list[tuple[int, float]] = []
        v19 = _finite(_get(doc, "axes", name, "direction", "arm_iddelta", "mean_cos_headline"))
        if v19 is not None:
            pairs.append((19, v19))
        lt = _get(doc, "axes", name, "layer_twins")
        if isinstance(lt, dict):
            for layer_key, blob in lt.items():
                v = _finite(_get(blob, "arm_iddelta_mean_cos_headline"))
                if v is not None:
                    pairs.append((int(layer_key), v))
        if len(pairs) >= 2:
            pairs.sort()
            series.append((name, [p[0] for p in pairs], [p[1] for p in pairs]))
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    cols = dict(zip([s[0] for s in series], paper_palette_blog(max(1, len(series))), strict=False))
    for name, ks, vs in series:
        ax.plot(ks, vs, marker="o", ms=4, color=cols[name], label=axis_label(name))
    ax.set_xticks(sorted({k for _, ks, _ in series for k in ks}))
    ax.set_xlabel("layer")
    ax.set_ylabel("direction cosine, identity baseline")
    ax.set_title("Layer sensitivity (identity baseline only)", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig_expl_layer_twins", out_dir)


def fig_pooling_twin(doc: dict, rows: list[dict], out_dir: Path) -> str | None:
    """Answer-tail vs span-mean pooling: headline direction cosine per axis and arm."""
    axes_names = _axes_sorted(doc)
    pts: list[tuple[str, str, float, float]] = []
    for name in axes_names:
        for arm in ARMS:
            tail = _finite(_get(doc, "axes", name, "direction", arm, "mean_cos_headline"))
            span = _finite(_get(doc, "axes", name, "pooling_twin_span", arm, "mean_cos_headline"))
            if tail is not None and span is not None:
                pts.append((name, arm, tail, span))
    if not pts:
        return None
    colors = _arm_colors()
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for arm in ARMS:
        xs = [t for _, a, t, _ in pts if a == arm]
        ys_ = [s for _, a, _, s in pts if a == arm]
        if xs:
            ax.scatter(xs, ys_, s=26, color=colors[arm], label=ARM_LABELS[arm], alpha=0.85)
    lim = [-1.05, 1.05]
    ax.plot(lim, lim, color=paper_palette_role("neutral"), lw=0.8, alpha=0.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("direction cosine, answer-tail pooling")
    ax.set_ylabel("direction cosine, span-mean pooling")
    ax.set_title("Pooling twin: answer tail vs span mean", loc="left")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "fig_expl_pooling_twin", out_dir)


FIGURES = (
    fig_hero_axis_profile,
    fig_norm_scatter,
    fig_install_vs_swap,
    fig_identity_heatmaps,
    fig_cross_family_scatter,
    fig_edit_dose,
    fig_retrieval_curves,
    fig_flip_vs_para_text,
    fig_carrier_transfer,
    fig_arm_agreement,
    fig_ceiling_vs_cos,
    fig_query_form_dissociation,
    fig_layer_twins,
    fig_pooling_twin,
)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="dir holding minpair_delta.json + perpair.jsonl (the PE out-dir)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="figures parent dir (stems land under issue_2564/)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="consume the /tmp smoke out-root and write figures to /tmp",
    )
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0

    results_dir = args.results_dir
    out_dir = args.out_dir
    if args.smoke:
        if args.results_dir == DEFAULT_RESULTS_DIR:
            results_dir = SMOKE_RESULTS_DIR
        if args.out_dir == DEFAULT_OUT_DIR:
            out_dir = SMOKE_ROOT / "figures"
        elif out_dir.resolve().is_relative_to((P2564.repo_root() / "figures").resolve()):
            # resolved-path compare (r2 blocker 4): the old string-startswith
            # guard missed absolute paths into the committed figures/ tree.
            raise SystemExit("--smoke must not write the committed figures/ tree")

    doc_path = results_dir / "minpair_delta.json"
    rows_path = results_dir / "perpair.jsonl"
    if not doc_path.is_file():
        raise SystemExit(f"missing {doc_path} — run scripts/issue2564_analysis.py first")
    doc = json.loads(doc_path.read_text())
    rows: list[dict] = []
    if rows_path.is_file():
        with rows_path.open(encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    else:
        print(f"[fig] perpair.jsonl missing at {rows_path}; per-pair figures will skip")

    set_paper_style("blog")
    wrote, skipped = [], []
    for fn in FIGURES:
        stem = fn(doc, rows, out_dir)
        if stem is None:
            skipped.append(fn.__name__)
            print(f"[fig] skip {fn.__name__}: input n/a in {doc_path.name}")
        else:
            wrote.append(stem)
            print(f"[fig] wrote {out_dir}/{STEM_PREFIX}/{stem}.png")
    print(f"[phase=figures] wrote={len(wrote)} skipped={len(skipped)} out={out_dir}")
    if not wrote:
        raise SystemExit("figures phase produced ZERO figures — inputs empty or malformed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
