"""Task #2546 P6 figures — heroes 1-3, graded-necessity backing panel, exploratory dump.

Reads the COMMITTED fit-driver artifacts (``eval_results/issue_2546/out/**``,
written by ``issue2546_fit_cells.py`` / ``issue2546_gen_capture.py`` /
``issue2546_n1m_read.py``) and renders the plan §6 figure set into
``figures/issue_2546/`` via the /paper-plots conventions
(``set_paper_style`` + ``savefig_paper`` sidecars).

Figure inventory (plan §6):
  hero1_cellgrid      Plot 7 grouped bars — 4 cells x 2 strata, R2 raw +
                      ceiling-normalized + content-identity lift (corpus pool /
                      same-template pool) panels, identity-baseline ticks;
                      arm 3 visually separated (different geometry).
  hero2_p8_ladder     arms 1-2 — Plot 8 E/F/G/H per-stratum bars, ladder tier
                      curves per corpus, sufficient-tier chart, operator panel
                      (direction-aware vs rotation-invariant-only, labeled).
  hero3_trajectory    R2 and content acc@1 vs t (11 points: p7_A endpoint,
                      9 interior t, p7_D endpoint), per stratum per arm, with
                      per-stratum noise ceilings + identity trajectory.
  backing_necessity   R2 / content lift vs graded corpus necessity rate at the
                      COMMITTED grain (per-corpus units x necessity class
                      rates; arm 3 uses the Qwen3 toggle labels). Finer
                      per-bin grains (GSM8K k-bins / ContextHub levels /
                      rescue-rate deciles) have no registry units — recorded
                      as concern ``backing-panel-grain-gap`` on #2546.
  exp_*               exploratory dump (layer sweeps, per-corpus heatmap,
                      matched-n companions, lambda diagnostics, p8_E baseline
                      decomposition, frozen n1m read, OOD transfer arms).

Missing REGISTERED input => FileNotFoundError naming the path (never a silent
skip); a present unit with ``status != ok`` (reported floor drop) renders as a
gap and is listed on stdout. ``--selftest`` renders EVERY figure class from a
synthetic JSON tree into a temp dir and asserts non-empty axes + sidecars.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before matplotlib/numpy: thread caps + headless env

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_blog,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

TASK = 2546
ARMS_ALL = (1, 2, 3)
HAS_PRE = {1: True, 2: True, 3: False}
ARM_TITLE = {
    1: "arm 1 — OpenThinker3-7B (pre: Qwen2.5-7B-Instruct)",
    2: "arm 2 — R1-Distill-Qwen-7B (pre: Qwen2.5-Math-7B)",
    3: "arm 3 — Qwen3-8B think on/off (different geometry)",
}
STRATA = ("does", "doesnt")
STRATUM_LABEL = {"does": "needs CoT", "doesnt": "no CoT needed"}
STRATUM_HATCH = {"does": "", "doesnt": "//"}
# Dedicated stratum hues (raw hexes: paper_palette_role names carry OTHER meanings
# elsewhere in this figure set — one color = one meaning).
STRATUM_COLOR = {"does": "#7b3294", "doesnt": "#008837"}
# Dedicated n1m read-pair hues (post/pre ctx-state reads); distinct from the
# ood identity-acc hue below and from every palette role used in sibling figures.
N1M_READ_COLOR = {"post_vc": "#2166ac", "pre_vc": "#d6604d"}
OOD_IDENTITY_ACC_COLOR = "#b35806"
CORPORA = ("gsm8k_train", "math", "contexthub", "mmlu")
CORPUS_LABEL = {
    "gsm8k_train": "GSM8K",
    "math": "MATH",
    "contexthub": "ContextHub",
    "mmlu": "MMLU",
    "pooled": "pooled",
}
P7_CELLS = ("p7_A", "p7_B", "p7_C", "p7_D")
P8_CELLS = ("p8_E", "p8_F", "p8_G", "p8_H")
# Plain-English cell names (no bare condition codes on reader-facing axes).
CELL_LABEL = {
    "p7_A": "ctx end → answer",
    "p7_Aoff": "ctx end → answer (think off)",
    "p7_B": "ctx end → CoT",
    "p7_C": "ctx end → full output",
    "p7_D": "CoT boundary → answer",
    "p8_E": "pre ctx → post answer",
    "p8_F": "post ctx → post answer",
    "p8_G": "pre ctx → pre answer",
    "p8_H": "pre ctx → post CoT",
}
_PAL8 = paper_palette(8)
CELL_COLOR = {
    "p7_A": _PAL8[0],
    "p7_Aoff": _PAL8[0],  # same map, think off — same hue, hatched below
    "p7_B": _PAL8[1],
    "p7_C": _PAL8[2],
    "p7_D": _PAL8[3],
    "p8_E": _PAL8[4],
    "p8_F": _PAL8[5],
    "p8_G": _PAL8[6],
    "p8_H": _PAL8[7],
}
_CPAL = paper_palette_blog(5)
CORPUS_COLOR = dict(zip(("pooled",) + CORPORA, _CPAL))
LADDER_TIER_LABELS = [
    "direct",
    "+ctx offset",
    "+ans offset",
    "+bias",
    "+scale",
    "+rotation",
    "reparam ctx",
    "reparam ans",
    "reparam both",
]

_DROPPED: list[str] = []  # non-ok units encountered (reported at exit)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def resolve_root(raw: Path) -> Path:
    """Results root: accept either <root> containing out/ or the out/ dir itself."""
    if (raw / "out").is_dir():
        return raw / "out"
    return raw


def _load(path: Path) -> dict:
    """Fail-loud registered-input read (never a silent skip)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"registered figure input missing: {path} — run the producing phase "
            "(P5 fits / P2a n1m read / gen necessity) and commit eval_results/issue_2546 first"
        )
    return json.loads(path.read_text())


def sweep_path(root: Path, cell: str, slug: str, arm: int) -> Path:
    return root / "cells" / f"{cell}__{slug}__a{arm}.json"


def read_sweep(root: Path, cell: str, slug: str, arm: int) -> dict | None:
    """Load a sweep unit; a reported non-ok status returns None (gap, listed)."""
    d = _load(sweep_path(root, cell, slug, arm))
    if d.get("status") != "ok":
        _DROPPED.append(f"{cell}__{slug}__a{arm}: status={d.get('status')}")
        return None
    return d


def read_reliability(root: Path, arm: int) -> dict[str, float | None]:
    """Per-stratum Spearman-Brown ceilings; None where not (yet) computable."""
    rel = _load(root / "reliability" / f"reliability__a{arm}.json")
    out: dict[str, float | None] = {}
    for s in STRATA:
        st = rel.get("per_stratum", {}).get(s) if rel.get("status") == "ok" else None
        if (
            st
            and st.get("status") == "ok"
            and np.isfinite(st.get("ceiling_spearman_brown", np.nan))
        ):
            out[s] = float(st["ceiling_spearman_brown"])
        else:
            out[s] = None
            _DROPPED.append(f"reliability__a{arm}[{s}]: ceiling unavailable")
    return out


def necessity_path(root: Path, arm: int) -> Path:
    name = "qwen3_toggle_labels.json" if arm == 3 else f"pair_necessity_a{arm}.json"
    return root / "necessity" / name


# ---------------------------------------------------------------------------
# Small plotting helpers
# ---------------------------------------------------------------------------


def _yerr(v: float, lo: float | None, hi: float | None) -> np.ndarray | None:
    """Non-negative errorbar OFFSETS from CI bounds (never raw bounds; clamped)."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(v)):
        return None
    return np.array([[max(0.0, v - float(lo))], [max(0.0, float(hi) - v)]])


def _boot_ci(d: dict | None) -> tuple[float | None, float | None]:
    if not d:
        return None, None
    return d.get("ci_lo"), d.get("ci_hi")


def _content_pool(d: dict | None, pool: str) -> dict | None:
    """knn_content euclidean pool block, or None (cot/out targets carry none)."""
    if not d:
        return None
    kc = d.get("knn_content")
    if not kc:
        return None
    blk = kc.get("euclidean", {}).get(pool)
    if not blk or "acc_at_1" not in blk:
        return None
    return blk


def _identity_tick(ax, x: float, y: float | None, w: float = 0.32) -> None:
    if y is None or not np.isfinite(y):
        return
    ax.plot([x - w / 2, x + w / 2], [y, y], color="black", lw=1.4, zorder=5)


def _assert_nonempty(fig) -> None:
    """Refuse to save a figure with no artists OR no finite plotted datum.

    Artist presence alone passes an all-NaN render (a bar of NaN height still
    creates a Rectangle patch) — require >=1 FINITE value across lines,
    patches, collections, or images (#1112: an empty figure was presented 3x).
    """

    def _finite(a) -> bool:
        for ln in a.lines:
            if ln.get_gid() == "refline":
                continue  # reference axhline/axvline — always finite, never a datum
            if np.isfinite(np.asarray(ln.get_ydata(), dtype=float)).any():
                return True
        for p in a.patches:
            h = getattr(p, "get_height", None)
            if h is not None and np.isfinite(float(h())):
                return True
        for c in a.collections:
            off = np.asarray(c.get_offsets(), dtype=float)
            if off.size and np.isfinite(off).any():
                return True
        return len(a.images) > 0

    n_artists = sum(
        len(a.lines) + len(a.patches) + len(a.collections) + len(a.images) for a in fig.axes
    )
    assert n_artists > 0, "figure rendered with no artists on any axes — refusing to save"
    assert any(_finite(a) for a in fig.axes), (
        "figure has artists but ZERO finite plotted data (all-NaN render) — refusing to save"
    )


def _save(fig, stem: str, out_dir: Path) -> None:
    _assert_nonempty(fig)
    paths = savefig_paper(fig, stem, dir=out_dir)
    plt.close(fig)
    print(f"[p6] wrote {paths.get('png', out_dir / (stem + '.png'))}", flush=True)


def _arm_cells(arm: int) -> list[str]:
    return ["p7_A", "p7_Aoff", "p7_B", "p7_C", "p7_D"] if arm == 3 else list(P7_CELLS)


def _bar(ax, x: float, y: float, cell: str, stratum: str, yerr=None) -> None:
    ax.bar(
        x,
        y,
        width=0.34,
        color=CELL_COLOR[cell],
        hatch=STRATUM_HATCH[stratum],
        alpha=0.55 if cell == "p7_Aoff" else 1.0,
        edgecolor="white",
        yerr=yerr,
        error_kw={"ecolor": "0.25", "lw": 1.0, "capsize": 2},
    )


# ---------------------------------------------------------------------------
# Hero 1 — Plot 7 grouped bars (cells x strata; 4 metric rows x arms)
# ---------------------------------------------------------------------------


def fig_hero1(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    ncol = len(arms)
    # spacer column before arm 3 (different geometry — visually separated)
    widths = []
    col_of: dict[int, int] = {}
    for a in arms:
        if a == 3 and widths:
            widths.append(0.12)
        col_of[a] = len(widths)
        widths.append(1.25 if a == 3 else 1.0)
    fig = plt.figure(figsize=(4.2 * ncol + 1.5, 11.0))
    gs = fig.add_gridspec(4, len(widths), width_ratios=widths, hspace=0.45, wspace=0.25)
    row_label = [
        "held-out R² (headline layer)",
        "R² / split-half ceiling",
        "answer-identity acc@1 − chance\n(corpus pool)",
        "answer-identity acc@1 − chance\n(same-template pool)",
    ]
    for a in arms:
        cells = _arm_cells(a)
        ceil = read_reliability(root, a)
        units = {(c, s): read_sweep(root, c, s, a) for c in cells for s in STRATA}
        xs = {c: float(i) for i, c in enumerate(cells)}
        for row in range(4):
            ax = fig.add_subplot(gs[row, col_of[a]])
            any_ceiling = any(v is not None for v in ceil.values())
            for c in cells:
                for j, s in enumerate(STRATA):
                    d = units[(c, s)]
                    if d is None:
                        continue
                    x = xs[c] + (j - 0.5) * 0.38
                    if row == 0:
                        v = float(d["r2_headline"])
                        lo, hi = _boot_ci(d.get("r2_headline_bootstrap"))
                        _bar(ax, x, v, c, s, yerr=_yerr(v, lo, hi))
                        idb = d.get("identity_bias_r2", {}).get(str(d["headline_layer"]))
                        _identity_tick(ax, x, idb)
                    elif row == 1:
                        cv = ceil.get(s)
                        if cv is None or cv <= 0:
                            continue
                        _bar(ax, x, float(d["r2_headline"]) / cv, c, s)
                    else:
                        pool = "corpus_pool" if row == 2 else "same_template_pool"
                        blk = _content_pool(d, pool)
                        if blk is None:
                            continue  # cot/out targets: content identity undefined
                        v = float(blk.get("lift", blk["acc_at_1"] - blk.get("chance_mean", 0.0)))
                        _bar(
                            ax,
                            x,
                            v,
                            c,
                            s,
                            yerr=_yerr(v, blk.get("lift_ci_lo"), blk.get("lift_ci_hi")),
                        )
                        if row == 2:
                            ki = d.get("knn_identity", {}).get("euclidean", {})
                            acc = ki.get("acc_at_k", {}).get("1")
                            ch = ki.get("chance_at_k", {}).get("1")
                            if acc is not None and ch is not None:
                                _identity_tick(ax, x, float(acc) - float(ch))
            ax.set_xticks([xs[c] for c in cells])
            ax.set_xticklabels([CELL_LABEL[c] for c in cells], rotation=30, ha="right", fontsize=7)
            ax.axhline(0.0, color="0.7", lw=0.8, gid="refline")
            if col_of[a] == 0:
                ax.set_ylabel(row_label[row], fontsize=8)
            title = ARM_TITLE[a] if row == 0 else ""
            if row == 1 and not any_ceiling:
                title = "N/A — split-half ceiling missing"
            if title:
                ax.set_title(title, fontsize=8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="0.55", hatch=STRATUM_HATCH[s], edgecolor="white")
        for s in STRATA
    ]
    handles.append(plt.Line2D([0], [0], color="black", lw=1.4))
    fig.legend(
        handles,
        [STRATUM_LABEL[s] for s in STRATA] + ["identity(+bias) baseline"],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    _save(fig, "hero1_cellgrid", out_dir)


# ---------------------------------------------------------------------------
# Hero 2 — Plot 8 bars + ladder + sufficient tier + operator (arms 1-2)
# ---------------------------------------------------------------------------


def fig_hero2(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in arms if HAS_PRE.get(a))
    if not arms:
        print("[p6] hero2 skipped: no pre-model arm requested", flush=True)
        return
    fig, axes = plt.subplots(4, len(arms), figsize=(5.4 * len(arms), 13.0), squeeze=False)
    for ci, a in enumerate(arms):
        # row 0: p8 cells x strata
        ax = axes[0][ci]
        for i, c in enumerate(P8_CELLS):
            for j, s in enumerate(STRATA):
                d = read_sweep(root, c, s, a)
                if d is None:
                    continue
                v = float(d["r2_headline"])
                lo, hi = _boot_ci(d.get("r2_headline_bootstrap"))
                _bar(ax, i + (j - 0.5) * 0.38, v, c, s, yerr=_yerr(v, lo, hi))
                _identity_tick(
                    ax,
                    i + (j - 0.5) * 0.38,
                    d.get("identity_bias_r2", {}).get(str(d["headline_layer"])),
                )
        ax.set_xticks(range(len(P8_CELLS)))
        ax.set_xticklabels([CELL_LABEL[c] for c in P8_CELLS], rotation=30, ha="right", fontsize=7)
        ax.set_title(ARM_TITLE[a], fontsize=8)
        if ci == 0:
            ax.set_ylabel("held-out R² (headline layer)", fontsize=8)
        # rows 1-2: ladder tier curves + sufficient tier per corpus
        ax1, ax2 = axes[1][ci], axes[2][ci]
        for slug in ("pooled",) + CORPORA:
            d = _load(root / "ladder" / f"ladder__{slug}__a{a}.json")
            if d.get("status") != "ok":
                _DROPPED.append(f"ladder__{slug}__a{a}: status={d.get('status')}")
                continue
            tiers = [d["tiers_r2"][t] for t in d["tier_names"]]
            col = CORPUS_COLOR[slug]
            ax1.plot(range(9), tiers, marker="o", ms=3, color=col, label=CORPUS_LABEL[slug])
            ref, band = float(d["within_post_reference_r2"]), float(d["band_value"])
            if slug == "pooled":
                ax1.axhspan(ref - band, ref, color="0.75", alpha=0.25, zorder=0)
                ax1.axhline(ref, color="0.4", lw=0.9, ls=":", gid="refline")
            st = d.get("sufficient_tier")
            if st is not None:
                ax1.plot([st], [tiers[st]], marker="*", ms=11, color=col, zorder=6)
                ax2.errorbar(
                    CORPUS_LABEL[slug],
                    st,
                    yerr=None,
                    marker="*",
                    ms=12,
                    color=col,
                    ls="none",
                )
            else:
                _DROPPED.append(f"ladder__{slug}__a{a}: no sufficient tier within band")
        ax1.set_xticks(range(9))
        ax1.set_xticklabels(LADDER_TIER_LABELS, rotation=35, ha="right", fontsize=7)
        if ci == 0:
            ax1.set_ylabel("pre→post map R² at tier", fontsize=8)
            ax2.set_ylabel("sufficient tier\n(within elicitation band)", fontsize=8)
        ax1.legend(fontsize=7, ncol=2)
        ax2.set_ylim(-0.5, 8.5)
        ax2.set_yticks(range(9))
        ax2.set_yticklabels(LADDER_TIER_LABELS, fontsize=7)
        ax2.grid(axis="y", lw=0.3, alpha=0.5)
        # row 3: operator comparison (status-guarded: a non-ok unit is a GAP, not a crash)
        ax3 = axes[3][ci]
        op = _load(root / "ladder" / f"operator_comparison__a{a}.json")
        if op.get("status") != "ok":
            _DROPPED.append(f"operator_comparison__a{a}: status={op.get('status')}")
            ax3.set_axis_off()
        else:
            raw = op["direction_aware"]["raw_cosine_with_rotation_null"]
            null = raw["rotation_null"]
            ax3.bar(0, float(raw["raw_cosine"]), width=0.5, color=paper_palette_role("primary"))
            nm, ns = float(null["null_mean"]), float(null["null_std"])
            # symmetric +-2*sigma band: the helper emits null_mean/std/p975 only (no
            # p025), so a p97.5 upper + 2-sigma lower would mix two interval
            # definitions in one bar — draw one coherent symmetric interval.
            ax3.errorbar(
                0.45,
                nm,
                yerr=_yerr(nm, nm - 2 * ns, nm + 2 * ns),
                marker="_",
                ms=14,
                color="0.35",
                ls="none",
                capsize=3,
            )
            ax3.bar(
                1.2,
                float(op["rotation_invariant_only"]["spectrum_cosine"]),
                width=0.5,
                color=paper_palette_role("neutral"),
                hatch="..",
                edgecolor="white",
            )
            ax3.set_xticks([0, 0.45, 1.2])
            ax3.set_xticklabels(
                [
                    "raw cosine\n(direction-aware)",
                    "rotation null\n(mean ± 2σ)",
                    "spectrum cosine\n(rotation-invariant only;\ncannot support same-operator)",
                ],
                fontsize=7,
            )
            if ci == 0:
                ax3.set_ylabel("operator similarity\n(pre-map vs post-map)", fontsize=8)
    _save(fig, "hero2_p8_ladder", out_dir)


# ---------------------------------------------------------------------------
# Hero 3 — trajectory (11 points: p7_A, t10..t90, p7_D) per stratum per arm
# ---------------------------------------------------------------------------


def _traj_series(root: Path, arm: int, stratum: str) -> dict | None:
    traj = _load(root / "cells" / f"p7_traj__a{arm}.json")
    st = traj.get("strata", {}).get(stratum)
    if traj.get("status") != "ok" or not st or st.get("status") != "ok":
        _DROPPED.append(f"p7_traj__a{arm}[{stratum}]: status not ok")
        return None
    a_cell = read_sweep(root, "p7_A", stratum, arm)
    d_cell = read_sweep(root, "p7_D", stratum, arm)
    t_grid = [float(t) for t in traj["t_grid"]]
    xs, r2, r2_lo, r2_hi, idb, acc, chance = [], [], [], [], [], [], []

    def _push_cell(x: float, d: dict | None) -> None:
        if d is None:
            return
        xs.append(x)
        r2.append(float(d["r2_headline"]))
        lo, hi = _boot_ci(d.get("r2_headline_bootstrap"))
        r2_lo.append(lo)
        r2_hi.append(hi)
        idb.append(d.get("identity_bias_r2", {}).get(str(d["headline_layer"])))
        blk = _content_pool(d, "corpus_pool")
        acc.append(blk["acc_at_1"] if blk else np.nan)
        chance.append(blk.get("chance_mean", np.nan) if blk else np.nan)

    _push_cell(0.0, a_cell)
    for t in t_grid:
        key = f"t{int(round(t * 100))}"
        pt = st["per_t"][key]
        xs.append(t)
        r2.append(float(pt["r2_headline"]))
        lo, hi = _boot_ci(pt.get("r2_headline_bootstrap"))
        r2_lo.append(lo)
        r2_hi.append(hi)
        idb.append(pt.get("identity_bias_r2_headline"))
        kc = pt.get("knn_content_euclidean", {}).get("corpus_pool", {})
        acc.append(kc.get("acc_at_1", np.nan))
        chance.append(kc.get("chance_mean", np.nan))
    _push_cell(1.0, d_cell)
    return {
        "x": xs,
        "r2": r2,
        "r2_lo": r2_lo,
        "r2_hi": r2_hi,
        "idb": idb,
        "acc": acc,
        "chance": chance,
    }


def fig_hero3(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    fig, axes = plt.subplots(2, len(arms), figsize=(4.6 * len(arms), 7.2), squeeze=False)
    scol = STRATUM_COLOR
    for ci, a in enumerate(arms):
        ceil = read_reliability(root, a)
        ax_r2, ax_acc = axes[0][ci], axes[1][ci]
        for s in STRATA:
            ser = _traj_series(root, a, s)
            if ser is None:
                continue
            x = np.asarray(ser["x"], dtype=float)
            r2 = np.asarray(ser["r2"], dtype=float)
            lo = np.array([np.nan if v is None else v for v in ser["r2_lo"]], dtype=float)
            hi = np.array([np.nan if v is None else v for v in ser["r2_hi"]], dtype=float)
            e_lo = np.maximum(0.0, np.nan_to_num(r2 - lo, nan=0.0))
            e_hi = np.maximum(0.0, np.nan_to_num(hi - r2, nan=0.0))
            ax_r2.errorbar(
                x,
                r2,
                yerr=np.vstack([e_lo, e_hi]),
                marker="o",
                ms=3.5,
                color=scol[s],
                label=STRATUM_LABEL[s],
                capsize=2,
                lw=1.4,
            )
            idb = np.array([np.nan if v is None else v for v in ser["idb"]], dtype=float)
            ax_r2.plot(x, idb, ls=":", color=scol[s], alpha=0.7, lw=1.0)
            if ceil.get(s) is not None:
                ax_r2.axhline(ceil[s], ls="--", color=scol[s], alpha=0.6, lw=1.0, gid="refline")
            ax_acc.plot(
                x, ser["acc"], marker="o", ms=3.5, color=scol[s], lw=1.4, label=STRATUM_LABEL[s]
            )
            ax_acc.plot(x, ser["chance"], ls=":", color=scol[s], alpha=0.7, lw=1.0)
        ax_r2.set_title(ARM_TITLE[a], fontsize=8)
        ax_acc.set_xlabel("CoT-content fraction t (0 = ctx end / p7_A, 1 = CoT boundary / p7_D)")
        if ci == 0:
            ax_r2.set_ylabel(
                "held-out R² (solid) / identity+bias (dotted)\nsplit-half ceiling (dashed)",
                fontsize=8,
            )
            ax_acc.set_ylabel(
                "answer-identity acc@1, corpus pool\n(solid; dotted = chance)", fontsize=8
            )
        ax_r2.legend(fontsize=7)
    _save(fig, "hero3_trajectory", out_dir)


# ---------------------------------------------------------------------------
# Backing panel — metric vs graded necessity (committed per-corpus grain)
# ---------------------------------------------------------------------------


def _necessity_rates(root: Path, arm: int) -> dict[str, float]:
    d = _load(necessity_path(root, arm))
    rates: dict[str, float] = {}
    for corpus, sizes in d.get("class_sizes", {}).items():
        classified = sum(v for k, v in sizes.items() if k != "unknown")
        if classified > 0:
            rates[corpus] = sizes.get("necessary", 0) / classified
    return rates


def fig_backing_necessity(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    fig, axes = plt.subplots(2, len(arms), figsize=(4.4 * len(arms), 7.0), squeeze=False)
    for ci, a in enumerate(arms):
        rates = _necessity_rates(root, a)
        cells = _arm_cells(a)
        ax_r2, ax_lift = axes[0][ci], axes[1][ci]
        for c in cells:
            pts_r2, pts_lift = [], []
            for corpus in CORPORA:
                if corpus not in rates:
                    continue
                d = read_sweep(root, c, corpus, a)
                if d is None:
                    continue
                pts_r2.append((rates[corpus], float(d["r2_headline"]), corpus))
                blk = _content_pool(d, "corpus_pool")
                if blk is not None:
                    pts_lift.append((rates[corpus], float(blk["lift"]), corpus))
            for ax, pts in ((ax_r2, pts_r2), (ax_lift, pts_lift)):
                if not pts:
                    continue
                pts = sorted(pts)
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    marker="o",
                    ms=4,
                    color=CELL_COLOR[c],
                    ls="--" if c == "p7_Aoff" else "-",
                    label=CELL_LABEL[c],
                    lw=1.3,
                )
        label_src = "toggle-necessity rate" if a == 3 else "pair-necessity rate"
        ax_lift.set_xlabel(f"corpus {label_src} (necessary / classified)")
        ax_r2.set_title(ARM_TITLE[a], fontsize=8)
        if ci == 0:
            ax_r2.set_ylabel("held-out R² (headline layer)", fontsize=8)
            ax_lift.set_ylabel("answer-identity lift\n(acc@1 − chance, corpus pool)", fontsize=8)
        ax_r2.legend(fontsize=6)
    _save(fig, "backing_necessity", out_dir)


# ---------------------------------------------------------------------------
# Exploratory dump
# ---------------------------------------------------------------------------


def fig_exp_layer_sweep(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 3.6), squeeze=False)
    for ci, a in enumerate(arms):
        ax = axes[0][ci]
        hl = None
        for c in _arm_cells(a):
            for s in STRATA:
                d = read_sweep(root, c, s, a)
                if d is None:
                    continue
                hl = d["headline_layer"]
                ax.plot(
                    range(len(d["r2_per_layer"])),
                    d["r2_per_layer"],
                    color=CELL_COLOR[c],
                    ls="-" if s == "does" else ":",
                    lw=1.0,
                    alpha=0.5 if c == "p7_Aoff" else 0.9,
                    label=f"{CELL_LABEL[c]} ({STRATUM_LABEL[s]})",
                )
        if hl is not None:
            ax.axvline(hl, color="0.4", lw=0.8, ls="--", gid="refline")
        ax.set_title(ARM_TITLE[a], fontsize=8)
        ax.set_xlabel("layer")
        if ci == 0:
            ax.set_ylabel("held-out R² per layer", fontsize=8)
            ax.legend(fontsize=5, ncol=2)
    _save(fig, "exp_layer_sweep", out_dir)


def fig_exp_percorpus_heatmap(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.4), squeeze=False)
    im = None
    for ci, a in enumerate(arms):
        ax = axes[0][ci]
        cells = _arm_cells(a)
        mat = np.full((len(cells), len(CORPORA)), np.nan)
        for i, c in enumerate(cells):
            for j, corpus in enumerate(CORPORA):
                d = read_sweep(root, c, corpus, a)
                if d is not None:
                    mat[i, j] = float(d["r2_headline"])
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(CORPORA)))
        ax.set_xticklabels([CORPUS_LABEL[c] for c in CORPORA], rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([CELL_LABEL[c] for c in cells], fontsize=7)
        ax.set_title(ARM_TITLE[a], fontsize=8)
    if im is not None:
        fig.colorbar(im, ax=list(axes[0]), label="held-out R² (headline layer)", shrink=0.85)
    _save(fig, "exp_percorpus_heatmap", out_dir)


def fig_exp_matchn(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 3.6), squeeze=False)
    for ci, a in enumerate(arms):
        ax = axes[0][ci]
        x = 0
        ticks, tick_labels = [], []
        for c in ("p7_A", "p7_D"):
            for s in STRATA:
                full = read_sweep(root, c, s, a)
                comp = _load(sweep_path(root, c, f"{s}__matchn", a))
                if comp.get("status") != "ok":
                    _DROPPED.append(f"{c}__{s}__matchn__a{a}: status={comp.get('status')}")
                    comp = None
                if full is not None:
                    _bar(ax, x - 0.2, float(full["r2_headline"]), c, s)
                if comp is not None:
                    ax.bar(
                        x + 0.2,
                        float(comp["r2_headline"]),
                        width=0.34,
                        color=CELL_COLOR[c],
                        hatch=STRATUM_HATCH[s],
                        alpha=0.45,
                        edgecolor="0.3",
                    )
                ticks.append(x)
                tick_labels.append(f"{CELL_LABEL[c]}\n{STRATUM_LABEL[s]}")
                x += 1
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=6)
        ax.set_title(ARM_TITLE[a], fontsize=8)
        if ci == 0:
            ax.set_ylabel(
                "held-out R² — full n (solid)\nvs matched-n companion (faded)", fontsize=8
            )
    _save(fig, "exp_matchn_companions", out_dir)


def fig_exp_lambda_diag(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 3.2), squeeze=False)
    for ci, a in enumerate(arms):
        ax = axes[0][ci]
        vals: list[float] = []
        selector = None
        for c in _arm_cells(a):
            for s in STRATA:
                d = read_sweep(root, c, s, a)
                if d is None:
                    continue
                diag = d.get("lambda_diag", {})
                selector = diag.get("selector", selector)
                sel = diag.get("selected") or []
                for row in sel:
                    for v in row:
                        if v is not None and np.isfinite(v) and v > 0:
                            vals.append(np.log10(v))
        if vals:
            ax.hist(vals, bins=20, color=paper_palette_role("primary"))
        ax.set_xlabel(f"log10 selected λ ({selector or 'selector unknown'})")
        ax.set_title(ARM_TITLE[a], fontsize=8)
        if ci == 0:
            ax.set_ylabel("count (folds × layers × units)", fontsize=8)
    _save(fig, "exp_lambda_diagnostics", out_dir)


def fig_exp_p8e_baselines(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in arms if HAS_PRE.get(a))
    if not arms:
        return
    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.4), squeeze=False)
    names = ["ridge map", "identity + bias", "train-mean", "random projection"]
    for ci, a in enumerate(arms):
        ax = axes[0][ci]
        d = read_sweep(root, "p8_E", "pooled", a)
        if d is not None:
            hl = str(d["headline_layer"])
            vals = [
                float(d["r2_headline"]),
                d.get("identity_bias_r2", {}).get(hl, np.nan),
                d.get("mean_baseline_r2", {}).get(hl, np.nan),
                d.get("random_projection_r2", {}).get(hl, np.nan),
            ]
            # identity+bias renders BLACK to match the black identity ticks every
            # sibling figure draws (_identity_tick) — one visual identity per baseline.
            colors = [
                paper_palette_role("primary"),
                "black",
                paper_palette_role("neutral"),
                paper_palette_role("control"),
            ]
            for i, (v, col) in enumerate(zip(vals, colors)):
                if v is not None and np.isfinite(v):
                    ax.bar(i, v, width=0.6, color=col)
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        ax.axhline(0.0, color="0.7", lw=0.8, gid="refline")
        ax.set_title(f"{ARM_TITLE[a]}\n{CELL_LABEL['p8_E']} (pooled)", fontsize=8)
        if ci == 0:
            ax.set_ylabel("held-out R² (headline layer)", fontsize=8)
    _save(fig, "exp_p8E_baseline_decomposition", out_dir)


def fig_exp_n1m_read(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    if 1 not in arms:
        return
    d = _load(root / "n1m_read" / "gsm8k_test1319_read.json")
    reads = d["reads"]
    layers = sorted(reads["post_vc"].keys(), key=lambda k: int(k[1:]))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    w = 0.35
    for j, (name, label) in enumerate(
        (
            ("post_vc", "post-model ctx state (read A)"),
            ("pre_vc", "pre-model ctx state (read B)"),
        )
    ):
        xs = np.arange(len(layers)) + (j - 0.5) * w
        r2s = [reads[name][layer]["r2_agreement"] for layer in layers]
        ax0.bar(xs, r2s, width=w, color=N1M_READ_COLOR[name], label=label)
        for x, layer in zip(xs, layers):
            _identity_tick(ax0, x, reads[name][layer].get("identity_r2_agreement"), w=0.25)
    ax0.set_xticks(range(len(layers)))
    ax0.set_xticklabels([f"layer {k[1:]}" for k in layers])
    ax0.set_ylabel("R² agreement with banked v̂ target\n(frozen #779 n1m ridge)", fontsize=8)
    ax0.legend(fontsize=7)
    ax0.axhline(0.0, color="0.7", lw=0.8, gid="refline")
    for layer in layers:
        pkb = reads["post_vc"][layer].get("per_k_bin", {})
        bins = sorted(b for b, v in pkb.items() if "r2_agreement" in v)
        if bins:
            ax1.plot(
                bins,
                [pkb[b]["r2_agreement"] for b in bins],
                marker="o",
                ms=4,
                label=f"layer {layer[1:]}",
            )
    ax1.set_xlabel("GSM8K reasoning-steps bin (k)")
    ax1.set_ylabel("R² agreement per k-bin (post read)", fontsize=8)
    ax1.legend(fontsize=7)
    ax0.set_title("frozen n1m read — GSM8K test (arm 1)", fontsize=8)
    _save(fig, "exp_n1m_frozen_read", out_dir)


def fig_exp_ood(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    arms = tuple(a for a in ARMS_ALL if a in arms)
    ood_units = (
        ("ood_gsm8k", "GSM8K train → test"),
        ("ood_does2doesnt", "needs-CoT → no-CoT"),
        ("ood_doesnt2does", "no-CoT → needs-CoT"),
    )
    fig, axes = plt.subplots(1, len(arms), figsize=(4.4 * len(arms), 3.4), squeeze=False)
    for ci, a in enumerate(arms):
        ax = axes[0][ci]
        for i, (stem, label) in enumerate(ood_units):
            d = _load(root / "cells" / f"{stem}__a{a}.json")
            if d.get("status") != "ok":
                _DROPPED.append(f"{stem}__a{a}: status={d.get('status')}")
                continue
            ax.bar(
                i - 0.19, float(d["transfer_r2"]), width=0.36, color=paper_palette_role("primary")
            )
            knn = d.get("knn_identity", {}).get("euclidean", {})
            acc = knn.get("acc_at_k", {}).get("1")
            if acc is not None:
                ax.bar(i + 0.19, float(acc), width=0.36, color=OOD_IDENTITY_ACC_COLOR)
                ch = knn.get("chance_at_k", {}).get("1")
                if ch is not None:
                    _identity_tick(ax, i + 0.19, float(ch), w=0.3)
        ax.set_xticks(range(len(ood_units)))
        ax.set_xticklabels([lb for _, lb in ood_units], rotation=20, ha="right", fontsize=7)
        ax.axhline(0.0, color="0.7", lw=0.8, gid="refline")
        ax.set_title(ARM_TITLE[a], fontsize=8)
        if ci == 0:
            ax.set_ylabel("transfer R² (left) / identity acc@1\n(right; tick = chance)", fontsize=8)
    _save(fig, "exp_ood_transfer", out_dir)


FIGURES = (
    fig_hero1,
    fig_hero2,
    fig_hero3,
    fig_backing_necessity,
    fig_exp_layer_sweep,
    fig_exp_percorpus_heatmap,
    fig_exp_matchn,
    fig_exp_lambda_diag,
    fig_exp_p8e_baselines,
    fig_exp_n1m_read,
    fig_exp_ood,
)


def render_all(root: Path, out_dir: Path, arms: tuple[int, ...]) -> None:
    set_paper_style("blog")
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn in FIGURES:
        fn(root, out_dir, arms)
    if _DROPPED:
        print(f"[p6] {len(_DROPPED)} non-ok unit(s) rendered as gaps (never zero bars):")
        for line in _DROPPED:
            print(f"[p6]   {line}")


# ---------------------------------------------------------------------------
# Selftest — synthetic JSON tree exercising every figure class
# ---------------------------------------------------------------------------


def _syn_knn_content(rng) -> dict:
    def pool() -> dict:
        acc = float(rng.uniform(0.2, 0.8))
        ch = float(rng.uniform(0.05, 0.15))
        return {
            "n": 40,
            "acc_at_1": acc,
            "chance_mean": ch,
            "lift": acc - ch,
            "lift_ci_lo": acc - ch - 0.05,
            "lift_ci_hi": acc - ch + 0.05,
        }

    return {
        "euclidean": {
            "corpus_pool": pool(),
            "same_template_pool": {**pool(), "coverage": 0.6},
            "per_corpus": {},
        },
        "cosine": {"corpus_pool": pool(), "same_template_pool": pool(), "per_corpus": {}},
    }


def _syn_sweep(rng, arm: int, cell: str, subset: str, hl: int, n_layers: int) -> dict:
    r2 = float(rng.uniform(0.3, 0.7))
    has_content = cell in ("p7_A", "p7_Aoff", "p7_D", "p8_E", "p8_F", "p8_G")
    return {
        "status": "ok",
        "unit_id": f"{cell}__{subset}__a{arm}",
        "arm": arm,
        "cell": cell,
        "subset": subset,
        "n_rows": 120,
        "headline_layer": hl,
        "frozen_layers": [max(0, hl - 5), hl, min(n_layers - 1, hl + 7)],
        "r2_per_layer": [float(v) for v in rng.uniform(0.0, r2, size=n_layers)],
        "r2_headline": r2,
        "identity_bias_r2": {str(hl): r2 - 0.15},
        "mean_baseline_r2": {str(hl): 0.02},
        "random_projection_r2": {str(hl): 0.05},
        "knn_identity": {
            m: {
                "metric": m,
                "n": 120,
                "n_pool": 120,
                "acc_at_k": {"1": float(rng.uniform(0.2, 0.7))},
                "chance_at_k": {"1": 1.0 / 120},
            }
            for m in ("euclidean", "cosine")
        },
        "knn_content": _syn_knn_content(rng) if has_content else None,
        "r2_headline_bootstrap": {"draws": [], "ci_lo": r2 - 0.06, "ci_hi": r2 + 0.05},
        "lambda_diag": {
            "selected": [[10.0 ** float(rng.uniform(0, 5)) for _ in range(3)] for _ in range(5)],
            "selector": "inner-group-cv",
        },
        "r2_ceiling_normalized": None,
        "ceiling_status": "missing_reliability_capture",
    }


def write_selftest_tree(root: Path) -> None:
    """Synthetic committed-artifact tree covering every consumed schema."""
    rng = np.random.default_rng(2546)
    out = root / "out"
    for sub in ("cells", "ladder", "reliability", "necessity", "n1m_read"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    def dump(p: Path, d: dict) -> None:
        p.write_text(json.dumps(d))

    for arm in ARMS_ALL:
        hl, n_layers = (24, 36) if arm == 3 else (19, 28)
        cells = _arm_cells(arm) + (list(P8_CELLS) if HAS_PRE[arm] else [])
        for cell in cells:
            subsets = list(STRATA) + list(CORPORA)
            if cell.startswith("p8"):
                subsets.append("pooled")
            for subset in subsets:
                dump(
                    out / "cells" / f"{cell}__{subset}__a{arm}.json",
                    _syn_sweep(rng, arm, cell, subset, hl, n_layers),
                )
        for cell in ("p7_A", "p7_D"):
            for s in STRATA:
                dump(
                    out / "cells" / f"{cell}__{s}__matchn__a{arm}.json",
                    _syn_sweep(rng, arm, cell, f"{s}__matchn", hl, n_layers),
                )
        # trajectory
        per_t_tpl = {}
        for t in [round(0.1 * i, 1) for i in range(1, 10)]:
            r2 = float(rng.uniform(0.3, 0.7))
            per_t_tpl[f"t{int(round(t * 100))}"] = {
                "r2_headline": r2,
                "r2_headline_bootstrap": {"ci_lo": r2 - 0.05, "ci_hi": r2 + 0.05},
                "identity_bias_r2_headline": r2 - 0.12,
                "knn_content_euclidean": {
                    "corpus_pool": {"acc_at_1": float(rng.uniform(0.2, 0.6)), "chance_mean": 0.1}
                },
            }
        dump(
            out / "cells" / f"p7_traj__a{arm}.json",
            {
                "status": "ok",
                "arm": arm,
                "t_grid": [round(0.1 * i, 1) for i in range(1, 10)],
                "headline_layer": hl,
                "strata": {
                    s: {"status": "ok", "per_t": json.loads(json.dumps(per_t_tpl))} for s in STRATA
                },
            },
        )
        # reliability
        dump(
            out / "reliability" / f"reliability__a{arm}.json",
            {
                "status": "ok",
                "arm": arm,
                "per_stratum": {
                    s: {
                        "status": "ok",
                        "n_prompts": 30,
                        "split_half_r2": 0.8,
                        "ceiling_spearman_brown": 0.89,
                    }
                    for s in STRATA
                },
            },
        )
        # necessity
        dump(
            necessity_path(root if (root / "necessity").is_dir() else out, arm),
            {
                "arm": arm,
                "class_sizes": {
                    c: {
                        "necessary": int(rng.integers(10, 60)),
                        "both_correct": 40,
                        "both_wrong": 10,
                        ("rescued_by_no_think" if arm == 3 else "pre_only_correct"): 5,
                        "unknown": 3,
                    }
                    for c in CORPORA
                },
            },
        )
        # ood
        for stem in ("ood_gsm8k", "ood_does2doesnt", "ood_doesnt2does"):
            dump(
                out / "cells" / f"{stem}__a{arm}.json",
                {
                    "status": "ok",
                    "transfer_r2": float(rng.uniform(0.1, 0.5)),
                    "knn_identity": {
                        "euclidean": {
                            "acc_at_k": {"1": float(rng.uniform(0.1, 0.5))},
                            "chance_at_k": {"1": 0.01},
                        }
                    },
                },
            )
        if HAS_PRE[arm]:
            for slug in ("pooled",) + CORPORA:
                tiers = sorted(rng.uniform(0.1, 0.62, size=9).tolist())
                names = [
                    "t0_direct_transfer",
                    "t1_context_offset",
                    "t2_answer_offset",
                    "t3_bias_offset",
                    "t4_global_scaling",
                    "t5_mapping_rotation",
                    "t6_reparam_contexts",
                    "t7_reparam_answers",
                    "t8_reparam_both",
                ]
                dump(
                    out / "ladder" / f"ladder__{slug}__a{arm}.json",
                    {
                        "status": "ok",
                        "tier_names": names,
                        "tiers_r2": dict(zip(names, tiers)),
                        "within_post_reference_r2": 0.64,
                        "band_value": 0.02,
                        "sufficient_tier": 6,
                    },
                )
            dump(
                out / "ladder" / f"operator_comparison__a{arm}.json",
                {
                    "status": "ok",
                    "direction_aware": {
                        "raw_cosine_with_rotation_null": {
                            "raw_cosine": 0.42,
                            "rotation_null": {
                                "null_mean": 0.001,
                                "null_std": 0.004,
                                "null_p975": 0.009,
                            },
                        }
                    },
                    "rotation_invariant_only": {"spectrum_cosine": 0.97},
                },
            )

    # n1m read (arm 1)
    def _knn_block() -> dict:
        return {"euclidean": {"acc_at_k": {"1": 0.4}, "chance_at_k": {"1": 0.001}}}

    layer_row = {
        "n_rows": 1319,
        "r2_agreement": 0.55,
        "identity_r2_agreement": 0.31,
        "knn": _knn_block(),
        "chance_at_1": 1.0 / 1319,
        "per_k_bin": {
            "k2-3": {"n_rows": 300, "r2_agreement": 0.5},
            "k4-5": {"n_rows": 400, "r2_agreement": 0.55},
            "k6+": {"n_rows": 300, "r2_agreement": 0.6},
        },
    }
    dump(
        out / "n1m_read" / "gsm8k_test1319_read.json",
        {
            "reads": {
                "post_vc": {f"L{v}": json.loads(json.dumps(layer_row)) for v in (14, 19, 26)},
                "pre_vc": {f"L{v}": json.loads(json.dumps(layer_row)) for v in (14, 19, 26)},
            }
        },
    )


def run_selftest() -> int:
    with tempfile.TemporaryDirectory(prefix="i2546_figs_selftest_") as td:
        root = Path(td) / "results"
        out_dir = Path(td) / "figs"
        write_selftest_tree(root)
        render_all(resolve_root(root), out_dir, ARMS_ALL)
        expected = {
            "hero1_cellgrid",
            "hero2_p8_ladder",
            "hero3_trajectory",
            "backing_necessity",
            "exp_layer_sweep",
            "exp_percorpus_heatmap",
            "exp_matchn_companions",
            "exp_lambda_diagnostics",
            "exp_p8E_baseline_decomposition",
            "exp_n1m_frozen_read",
            "exp_ood_transfer",
        }
        for stem in sorted(expected):
            png = out_dir / f"{stem}.png"
            meta = out_dir / f"{stem}.meta.json"
            assert png.is_file() and png.stat().st_size > 0, f"selftest: missing/empty {png}"
            assert meta.is_file(), f"selftest: missing sidecar {meta}"
        # All-NaN refline rejection (r4; r3 blocker claimed-revision-fixtures-
        # absent (c)): a figure whose ONLY finite content is gid="refline"
        # reference lines must be REFUSED — reflines are excluded from the
        # finite-datum scan, so removing that exclusion (the defect direction:
        # the always-finite axhline would vacuously satisfy the guard) makes
        # this fixture fail by NOT raising.
        fig, ax = plt.subplots()
        ax.bar([0.0, 1.0], [float("nan"), float("nan")])
        ax.axhline(0.5).set_gid("refline")
        try:
            try:
                _assert_nonempty(fig)
            except AssertionError as e:
                assert "ZERO finite plotted data" in str(e), e
            else:
                raise AssertionError("all-NaN + refline-only figure passed _assert_nonempty")
        finally:
            plt.close(fig)
        # Control (guard not vacuous): ONE finite datum beside the refline passes.
        fig2, ax2 = plt.subplots()
        ax2.bar([0.0, 1.0], [float("nan"), 0.7])
        ax2.axhline(0.5).set_gid("refline")
        try:
            _assert_nonempty(fig2)
        finally:
            plt.close(fig2)
        print(
            f"[p6] SELFTEST PASS — {len(expected)} figure classes rendered with "
            "sidecars; all-NaN refline-only render refused"
        )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--results-root",
        default="eval_results/issue_2546",
        help="committed results root (containing out/ or being the out/ dir itself)",
    )
    ap.add_argument("--out-dir", default="figures/issue_2546")
    ap.add_argument("--arms", default="1,2,3", help="comma-separated arm ids")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[p6] import-check OK")
        return 0
    if args.selftest:
        return run_selftest()
    arms = tuple(int(v) for v in str(args.arms).split(",") if v.strip())
    assert arms and all(a in ARMS_ALL for a in arms), f"--arms invalid: {args.arms}"
    render_all(resolve_root(Path(args.results_root)), Path(args.out_dir), arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
