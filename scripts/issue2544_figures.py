#!/usr/bin/env python3
"""#2544 P5 figure driver — formation-curve figures from the committed eval JSONs.

Reads ``eval_results/issue_2544/**`` (the P4a/P4b registered outputs; override
with ``--eval-dir`` or ``EPM_ISSUE2544_EVAL_DIR`` for the smoke's diverted
tree). Writes ``figures/issue_2544/`` via ``savefig_paper`` (PNG + PDF + meta
sidecar). Conventions (/paper-plots, binding): axes + ticks + legend + panel
titles only — no ``fig.text`` caption blocks (facts live in the sidecar);
ONE COLOR = ONE MEANING across the whole set (module-level ``COLORS``); every
aggregate hero gets a per-unit companion with labeled points; Δ(T) panels
carry the over-window scope label in the panel title (§6 sliding-window
adjudication); every errorbar passes NON-NEGATIVE offsets (never CI bounds).

Figures (plan §6 items 1–5):
  fig1  HERO D(T) formation curve (raw + baseline-subtracted) + fig1b per-fold
  fig2  HERO decomposition (column / row / diagonal)          + fig2b per-band
  fig3  HERO retention (i→main by mode + adjacent)            + fig3b per-pair
  fig4  HERO Δ(T) k-shot substitution (+dose, +over-window)   + fig4b per-fold
  fig5* exploratory dump: per-class, 17-layer sweep, best-layer selection
        mass, kNN retrieval, λ*/dof selector, over-window lengths,
        trained-only sensitivity, native-vs-plain, robustness arms, ceilings.

RSS ≪ 16 GB: aggregate JSONs + one percell ``.npz`` at a time (plan §9 P5).
A figure with NO data series is SKIPPED with a printed reason — an empty
render is never saved; missing PRIMARY inputs raise (fail-loud).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps before numpy/matplotlib import (shared-VM rule)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import issue2544_common as C2  # noqa: E402  (env-order: MUST precede issue1902_common)
import issue1902_common as C1  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

# ── constants ────────────────────────────────────────────────────────────────

# Olmo-3 stage-1 tokens/step: B = 1,024 sequences × C = 4,096 tokens (plan §0.0
# induction-law block; r9 = step 1,413,814 → 5.93T, matching the plan's figure).
TOKENS_PER_STEP = 1024 * 4096
INDUCTION_TOKENS = 1.07e9  # predicted induction transition (plan §0.0, 2511.16893)

_PAL = paper_palette(8)  # the curated 8-colour palette (paper_palette warns past 8)
# One color = one meaning ACROSS every figure in this set. _PAL[0]/_PAL[1] are
# the primary/baseline role colours (diag/identity below), so the remaining
# meanings take _PAL[2..7] plus explicit off-palette hexes (IBM colours).
COLORS: dict[str, str] = {
    "diag": paper_palette_role("primary"),  # diagonal D(T) / headline R²
    "identity": paper_palette_role("baseline"),  # identity+bias baseline
    "null": "#9a9a9a",  # matched / shuffled nulls
    "ceiling": _PAL[2],  # split-half reliability ceilings
    "dtilde": _PAL[3],  # baseline-subtracted D̃
    "colC": _PAL[4],  # fixed-answer-text column cells
    "rowR": _PAL[5],  # fixed-weights row cells
    "direct": _PAL[6],  # transfer: direct
    "gl": _PAL[7],  # transfer: general-linear
    "orth": "#994455",  # transfer: orthogonal Procrustes
    "delta": "#DC267F",  # Δ(T) pooled
    "delta_ww": "#785EF0",  # Δ_ww within-window companion
    "delta_fa": "#994F00",  # Δ_FA full-attention companion
}
FLOOR_COLOR = "#444444"

# ── io helpers ───────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    """Fail-loud JSON read (primary inputs must exist)."""
    if not path.is_file():
        raise FileNotFoundError(f"required eval input missing: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _err(points: list[float], cis: list[Any]) -> np.ndarray:
    """CI bounds -> NON-NEGATIVE per-point offsets (xerr/yerr contract)."""
    lo: list[float] = []
    hi: list[float] = []
    for p, ci in zip(points, cis):
        ok = (
            isinstance(ci, (list, tuple))
            and len(ci) == 2
            and np.isfinite(p)
            and np.isfinite(ci[0])
            and np.isfinite(ci[1])
        )
        if ok:
            lo.append(max(0.0, float(p) - float(ci[0])))
            hi.append(max(0.0, float(ci[1]) - float(p)))
        else:
            lo.append(0.0)
            hi.append(0.0)
    return np.asarray([lo, hi])


def _fmt_tokens(t: float) -> str:
    if t >= 1e12:
        return f"{t / 1e12:.2g}T"
    if t >= 1e9:
        return f"{t / 1e9:.2g}B"
    return f"{t:.2g}"


def _rung_positions(rungs: list[str]) -> tuple[dict[str, float], list[str], float | None]:
    """x positions per rung: log10(pretraining tokens) for stage1-step>0 rungs;
    r0 pinned one slot left of the log zone; midtrain/final/post rungs appended
    categorically right (their per-stage token counts are not published).
    Returns (pos_by_rung, ticklabels aligned to rungs, log-zone left edge)."""
    branches = getattr(C1, "MODEL_BRANCHES", {}) or {}
    logx: dict[str, float] = {}
    for r in rungs:
        m = re.fullmatch(r"stage1-step(\d+)", str(branches.get(r) or ""))
        if m and int(m.group(1)) > 0:
            logx[r] = float(np.log10(int(m.group(1)) * TOKENS_PER_STEP))
    pos: dict[str, float] = {}
    labels: list[str] = []
    if logx:
        left = min(logx.values())
        right = max(logx.values())
        cat = left - 1.0
        k_right = 0
        for r in rungs:
            if r in logx:
                pos[r] = logx[r]
                labels.append(f"{r}\n{_fmt_tokens(10 ** logx[r])}")
            elif r == "r0":
                pos[r] = cat
                labels.append("r0\n(init)")
            else:
                k_right += 1
                pos[r] = right + 0.8 * k_right
                labels.append(r)
        return pos, labels, left
    for k, r in enumerate(rungs):
        pos[r] = float(k)
        labels.append(r)
    return pos, labels, None


def _apply_rung_axis(ax: plt.Axes, rungs: list[str], pos: dict[str, float], labels) -> None:
    ax.set_xticks([pos[r] for r in rungs])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("checkpoint (pretraining tokens where published)")


def _label_points(ax: plt.Axes, xs, ys, texts, color: str) -> None:
    """Per-unit data labels (the labeled-points companion convention)."""
    for x, y, t in zip(xs, ys, texts):
        if np.isfinite(y):
            ax.annotate(
                t, (x, y), fontsize=5.5, color=color, xytext=(2, 2), textcoords="offset points"
            )


# ── input bundle ─────────────────────────────────────────────────────────────


class Inputs:
    """Loaded eval artifacts (aggregates in RAM; percell npz read on demand)."""

    def __init__(self, eval_dir: Path):
        self.eval_dir = eval_dir
        fits = eval_dir / "fits"
        self.diag = _read_json(fits / "diag_curve.json")
        self.cross = _read_json(fits / "cross_cells.json")
        self.kshot = _read_json(fits / "kshot_curve.json")
        self.retention = _read_json(eval_dir / "transfer" / "retention_matrix.json")
        self.sweep = _read_json(fits / "layer_sweep.json")
        nm = fits / "layer_null_matrix.npz"
        if not nm.is_file():
            raise FileNotFoundError(f"required eval input missing: {nm}")
        self.nullmat = np.load(nm, allow_pickle=False)
        self.percell = fits / "percell"
        self.rungs: list[str] = list(self.diag["rung_order"])
        self.per_rung: dict[str, Any] = self.diag["per_rung"]
        if not self.per_rung:
            raise RuntimeError("diag_curve.json has an EMPTY per_rung — refusing to plot")
        self.layer_star = int(self.diag["layer_star"])
        self.layers17 = [int(x) for x in self.nullmat["layers"]]
        self.li_star = self.layers17.index(self.layer_star)
        self.band6 = [int(x) for x in self.kshot["band_b6"]]
        self.pos, self.ticklabels, self.log_left = _rung_positions(self.rungs)
        # v8 censoring family (absent on pre-v8 eval trees -> strips/panels skip)
        self.censor: dict[str, Any] = self.diag.get("censoring_sensitivity") or {}
        self.trunc_table: dict[str, Any] = self.censor.get("per_rung_truncation") or {}

    def fold_star_r2(self, rung: str, cell: str = "diag0") -> dict[int, float]:
        """Per-fold pooled R² at layer* from percell shards (one npz at a time)."""
        out: dict[int, float] = {}
        band_cell = cell != "diag0"
        idx = self.band6.index(self.layer_star) if band_cell else self.li_star
        for p in sorted(self.percell.glob(f"{cell}_{rung}_f*.npz")):
            m = re.search(r"_f(\d+)\.npz$", p.name)
            if not m:
                continue
            d = np.load(p)
            res = float(np.nansum(d["ss_res"][idx]))
            tot = float(np.nansum(d["ss_tot"][idx]))
            out[int(m.group(1))] = 1.0 - res / tot if tot > 0 else float("nan")
        return out


# ── figures ──────────────────────────────────────────────────────────────────


def fig1_formation(inp: Inputs, out_dir: Path) -> str | None:
    """HERO: D(T) raw + baseline-subtracted, with identity band, shuffled null,
    ceilings, formation floors, induction-transition marker (plan §6 item 1)."""
    rungs, pr = inp.rungs, inp.per_rung
    r2 = [pr[r]["r2_star"] for r in rungs]
    ident = [pr[r]["identity_r2_star"] for r in rungs]
    ceil = [(pr[r].get("ceiling_0shot") or {}).get("spearman_brown") for r in rungs]
    null_max = [(pr[r].get("shuffle_null_r2") or {}).get("max") for r in rungs]
    dt = [pr[r]["dtilde"] for r in rungs]
    xs = [inp.pos[r] for r in rungs]

    # v8: truncation-censoring strip under the hero panels (reported axis —
    # truncation never filters rows under rep-only-v2 eligibility).
    strip_rungs = [r for r in rungs if r in inp.trunc_table]
    ax_s = None
    if strip_rungs:
        fig = plt.figure(figsize=(11.5, 5.6), layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.0])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_s = fig.add_subplot(gs[1, :])
    else:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    ax_a.errorbar(
        xs,
        r2,
        yerr=_err(r2, [pr[r]["ci_frozen"] for r in rungs]),
        fmt="o-",
        color=COLORS["diag"],
        label="diagonal OOF R² (D)",
        capsize=2,
    )
    ax_a.errorbar(
        xs,
        ident,
        yerr=_err(ident, [pr[r]["identity_ci"] for r in rungs]),
        fmt="s--",
        color=COLORS["identity"],
        label="identity+bias baseline",
        capsize=2,
    )
    if any(v is not None and np.isfinite(v) for v in np.asarray(ceil, dtype=float)):
        ax_a.plot(
            xs,
            np.asarray(ceil, dtype=float),
            "^:",
            color=COLORS["ceiling"],
            label="split-half ceiling (SB)",
        )
    if any(v is not None for v in null_max):
        ax_a.plot(
            xs,
            np.asarray(null_max, dtype=float),
            "v",
            color=COLORS["null"],
            label="shuffled-pairing null (max)",
        )
    if inp.log_left is not None:
        ind = float(np.log10(INDUCTION_TOKENS))
        ax_a.axvspan(inp.pos["r0"] + 0.15, inp.log_left - 0.02, color="0.85", alpha=0.5, zorder=0)
        if ind < inp.log_left:
            ax_a.axvline(
                ind,
                color="0.4",
                linestyle=":",
                linewidth=1,
                label="predicted induction transition (~1.07B)",
            )
    _apply_rung_axis(ax_a, rungs, inp.pos, inp.ticklabels)
    ax_a.set_ylabel("pooled OOF R² at layer*")
    ax_a.set_title(f"D(T): 0-shot diagonal at layer* = {inp.layer_star}")
    ax_a.legend(fontsize=7)

    ax_b.errorbar(
        xs,
        dt,
        yerr=_err(dt, [pr[r]["dtilde_ci"] for r in rungs]),
        fmt="o-",
        color=COLORS["dtilde"],
        label="D̃ (95% CI, frozen layer*)",
        capsize=2,
    )
    ax_b.errorbar(
        xs,
        dt,
        yerr=_err(dt, [pr[r]["dtilde_ci_selection_inherited"] for r in rungs]),
        fmt="none",
        ecolor=COLORS["dtilde"],
        alpha=0.35,
        elinewidth=3,
        label="selection-inherited CI",
    )
    ax_b.axhline(0.10, color=FLOOR_COLOR, linewidth=1, label="formation floor 0.10")
    for f in (0.05, 0.15):
        ax_b.axhline(f, color=FLOOR_COLOR, linewidth=0.7, linestyle=":")
    _apply_rung_axis(ax_b, rungs, inp.pos, inp.ticklabels)
    ax_b.set_ylabel("D̃ = D − identity+bias R²")
    ax_b.set_title("baseline-subtracted formation curve")
    ax_b.legend(fontsize=7)
    if ax_s is not None:
        sx = [inp.pos[r] for r in strip_rungs]
        ax_s.plot(
            sx,
            [inp.trunc_table[r]["gen0"] for r in strip_rungs],
            "o-",
            color="0.35",
            label="gen0 (0-shot arm)",
        )
        ax_s.set_ylim(0, 1.02)
        ax_s.set_ylabel("truncation rate")
        ax_s.set_title("per-rung truncation-censoring rate (reported, never filtered)")
        _apply_rung_axis(ax_s, rungs, inp.pos, inp.ticklabels)
        ax_s.legend(fontsize=7)
    savefig_paper(fig, "fig1_formation_curve", dir=out_dir)
    plt.close(fig)
    return "fig1_formation_curve"


def fig1b_formation_perfold(inp: Inputs, out_dir: Path) -> str | None:
    """Companion: per-fold star-layer R² points (labeled) under the pooled curve."""
    fig, ax = plt.subplots(figsize=(8.5, 4.0), layout="constrained")
    plotted = False
    for r in inp.rungs:
        folds = inp.fold_star_r2(r)
        if not folds:
            continue
        plotted = True
        xs = [inp.pos[r]] * len(folds)
        ys = list(folds.values())
        ax.scatter(xs, ys, s=14, color=COLORS["diag"], alpha=0.6)
        _label_points(ax, xs, ys, [f"f{k}" for k in folds], COLORS["diag"])
        ax.plot(inp.pos[r], inp.per_rung[r]["r2_star"], marker="_", markersize=16, color="black")
    if not plotted:
        print("[figures] skip fig1b: no percell diag0 shards", flush=True)
        plt.close(fig)
        return None
    _apply_rung_axis(ax, inp.rungs, inp.pos, inp.ticklabels)
    ax.set_ylabel("per-fold OOF R² at layer*")
    ax.set_title("per-fold diagonal R² (black tick = pooled)")
    savefig_paper(fig, "fig1b_formation_perfold", dir=out_dir)
    plt.close(fig)
    return "fig1b_formation_perfold"


def fig2_decomposition(inp: Inputs, out_dir: Path) -> str | None:
    """HERO: fixed-answer-text column vs fixed-weights row vs diagonal (§6 item 2)."""
    cells = inp.cross["cells"]
    col, row = cells.get("colC", {}), cells.get("rowR", {})
    if not col and not row:
        print("[figures] skip fig2: no cross cells", flush=True)
        return None
    rungs = inp.rungs
    xs = [inp.pos[r] for r in rungs]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    diag = [cells["diag_ref"][r]["r2_star"] for r in rungs]
    ax.plot(xs, diag, "o-", color=COLORS["diag"], label="diagonal (m = s)")
    for series, name, label in (
        (col, "colC", "fixed answer text (column, s = main): representation change"),
        (row, "rowR", "fixed weights (row, m = main): answer-distribution change"),
    ):
        pts = [(inp.pos[r], series[r]["r2_star"], series[r]["ci"]) for r in rungs if r in series]
        if not pts:
            continue
        px, py, pc = zip(*pts)
        ax.errorbar(
            px,
            py,
            yerr=_err(list(py), list(pc)),
            fmt="s--" if name == "colC" else "D-.",
            color=COLORS[name],
            label=label,
            capsize=2,
        )
    _apply_rung_axis(ax, rungs, inp.pos, inp.ticklabels)
    ax.set_ylabel("pooled OOF R² at layer*")
    ax.set_title("decomposition: representation vs answer-distribution change")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig2_decomposition", dir=out_dir)
    plt.close(fig)
    return "fig2_decomposition"


def fig2b_decomposition_band(inp: Inputs, out_dir: Path) -> str | None:
    """Companion: per-cell per-band R² over B6 with the layer-type split marked."""
    cells = inp.cross["cells"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    plotted = False
    for kind, style in (("colC", "-"), ("rowR", "--")):
        for key, cell in sorted(cells.get(kind, {}).items()):
            band = cell.get("per_band_r2") or {}
            if not band:
                continue
            plotted = True
            ls = sorted(int(k) for k in band)
            ax.plot(ls, [band[str(x)] for x in ls], style, color=COLORS[kind], alpha=0.65)
            ax.annotate(
                f"{kind}:{key}",
                (ls[-1], band[str(ls[-1])]),
                fontsize=5.5,
                color=COLORS[kind],
                xytext=(3, 0),
                textcoords="offset points",
            )
    if not plotted:
        print("[figures] skip fig2b: no per-band cross reads", flush=True)
        plt.close(fig)
        return None
    split = inp.kshot.get("layer_type_split_b6") or {}
    for x in split.get("full_attention", []):
        ax.axvline(int(x), color="0.75", linestyle=":", linewidth=1)
    ax.axvline(inp.layer_star, color="black", linewidth=1, label=f"layer* = {inp.layer_star}")
    ax.set_xlabel("layer (dotted grey = full-attention layers in B6)")
    ax.set_ylabel("pooled OOF R²")
    ax.set_title("cross cells across the B6 band")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig2b_decomposition_band", dir=out_dir)
    plt.close(fig)
    return "fig2b_decomposition_band"


def fig3_retention(inp: Inputs, out_dir: Path) -> str | None:
    """HERO: ρ(i→main) by mode vs matched nulls, T_c; adjacent retention (§6 item 3)."""
    pairs = inp.retention["pairs"]
    to_main = {k.split("->")[0]: v for k, v in pairs.items() if k.endswith("->main")}
    if not to_main:
        print("[figures] skip fig3: no i->main transfer pairs", flush=True)
        return None
    q_main = inp.per_rung["main"]["r2_star"]
    rungs_i = [r for r in inp.rungs if r in to_main]
    xs = [inp.pos[r] for r in rungs_i]
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.5, 4.2), layout="constrained", gridspec_kw={"width_ratios": [3, 2]}
    )
    for mode, fmt in (("direct", "o-"), ("gl", "s--"), ("orth", "D-.")):
        pts = [to_main[r]["rho_vs_Q_jj"][mode] for r in rungs_i]
        ys = [p["rho"] if p["rho"] is not None else np.nan for p in pts]
        ax_a.errorbar(
            xs,
            ys,
            yerr=_err(ys, [p["ci"] for p in pts]),
            fmt=fmt,
            color=COLORS[mode],
            label=f"ρ {mode}(i→main)",
            capsize=2,
        )
    null_pts = {"shuffled_correspondence": [], "spectrum_matched": []}
    for r in rungs_i:
        for nk in null_pts:
            vals = to_main[r].get("null_r2", {}).get(nk, [])
            if vals and q_main:
                null_pts[nk].append((inp.pos[r], max(vals) / q_main))
    for nk, marker in (("shuffled_correspondence", "x"), ("spectrum_matched", "+")):
        if null_pts[nk]:
            nx, ny = zip(*null_pts[nk])
            ax_a.plot(nx, ny, marker, color=COLORS["null"], label=f"{nk} null (max)")
    tc = inp.retention.get("t_c", {}).get("point_rung")
    if tc is not None and tc in inp.pos:
        ax_a.axvline(
            inp.pos[tc],
            color="black",
            linestyle="-",
            linewidth=1,
            label=f"T_c = {tc} (ρ_orth ≥ 0.8 suffix)",
        )
    ax_a.axhline(0.8, color=FLOOR_COLOR, linewidth=0.7, linestyle=":")
    ax_a.axhline(0.5, color=FLOOR_COLOR, linewidth=0.7, linestyle=":")
    _apply_rung_axis(ax_a, rungs_i, inp.pos, [inp.ticklabels[inp.rungs.index(r)] for r in rungs_i])
    ax_a.set_ylabel("ρ = transfer R² / Q(main, main)")
    ax_a.set_title("retention onto the final base map")
    ax_a.legend(fontsize=6.5)

    adjacent = inp.retention.get("adjacent_transitions", [])
    adj = [(k, pairs[k]) for k in adjacent if k in pairs]
    if adj:
        ay = [p["rho_vs_Q_jj"]["gl"]["rho"] for _, p in adj]
        ay = [v if v is not None else np.nan for v in ay]
        ac = [p["rho_vs_Q_jj"]["gl"]["ci"] for _, p in adj]
        ax_b.errorbar(
            range(len(adj)),
            ay,
            yerr=_err(ay, ac),
            fmt="s--",
            color=COLORS["gl"],
            capsize=2,
            label="ρ gl (adjacent)",
        )
        ref = (inp.retention.get("reference") or {}).get("issue1902_median_adjacent_gl_retention")
        if ref:
            ax_b.axhline(
                float(ref),
                color="0.5",
                linestyle="--",
                linewidth=1,
                label=f"#1902 median adjacent ({ref})",
            )
        ax_b.set_xticks(range(len(adj)))
        ax_b.set_xticklabels([k.replace("->", "→") for k, _ in adj], rotation=60, fontsize=6)
        ax_b.set_ylabel("ρ gl")
        ax_b.set_title("adjacent-transition retention")
        ax_b.legend(fontsize=6.5)
    savefig_paper(fig, "fig3_retention", dir=out_dir)
    plt.close(fig)
    return "fig3_retention"


def fig3b_retention_perpair(inp: Inputs, out_dir: Path) -> str | None:
    """Companion: every transfer pair's ρ by mode, pair-labeled."""
    pairs = inp.retention["pairs"]
    if not pairs:
        print("[figures] skip fig3b: no transfer pairs", flush=True)
        return None
    keys = sorted(pairs)
    fig, ax = plt.subplots(figsize=(max(7.0, 0.28 * len(keys) + 2), 4.2), layout="constrained")
    for mode, marker in (("direct", "o"), ("gl", "s"), ("orth", "D")):
        ys = [pairs[k]["rho_vs_Q_jj"][mode]["rho"] for k in keys]
        ys = [v if v is not None else np.nan for v in ys]
        ax.scatter(range(len(keys)), ys, s=16, marker=marker, color=COLORS[mode], label=mode)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("->", "→") for k in keys], rotation=75, fontsize=5.5)
    ax.axhline(0.8, color=FLOOR_COLOR, linewidth=0.7, linestyle=":")
    ax.set_ylabel("ρ = transfer R² / Q(j, j)")
    ax.set_title("per-pair retention (all fitted pairs)")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig3b_retention_perpair", dir=out_dir)
    plt.close(fig)
    return "fig3b_retention_perpair"


def fig4_kshot(inp: Inputs, out_dir: Path) -> str | None:
    """HERO: Δ(T) with Δ_ww/Δ_FA companions + robustness band; dose panel;
    over-window fraction strip (§6 item 4). The pooled Δ panel title carries
    the sliding-window scope label (§6 adjudication)."""
    per = inp.kshot.get("per_rung") or {}
    if not per:
        print("[figures] skip fig4: no k-shot rungs", flush=True)
        return None
    rungs = [r for r in inp.rungs if r in per]
    xs = [inp.pos[r] for r in rungs]
    # v8: per-ARM truncation strip under the Δ panels (kshot_curve carries its
    # own dual-arm table; falls back to the diag table on older eval trees).
    trunc = inp.kshot.get("per_rung_truncation") or inp.trunc_table
    strip_rungs = [r for r in rungs if r in trunc]
    ax_s = None
    if strip_rungs:
        fig = plt.figure(figsize=(13.5, 5.6), layout="constrained")
        gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1.0], width_ratios=[3, 2, 2])
        axes = [fig.add_subplot(gs[0, k]) for k in range(3)]
        ax_s = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13.5, 4.2),
            layout="constrained",
            gridspec_kw={"width_ratios": [3, 2, 2]},
        )
    ax_a, ax_b, ax_c = axes
    dl = [per[r]["delta"] for r in rungs]
    ax_a.errorbar(
        xs,
        dl,
        yerr=_err(dl, [per[r]["delta_ci"] for r in rungs]),
        fmt="o-",
        color=COLORS["delta"],
        label="Δ pooled (4-shot − 0-shot)",
        capsize=2,
    )
    ww = [(inp.pos[r], per[r]["delta_ww"]) for r in rungs if "delta_ww" in per[r]]
    if ww:
        wx = [x for x, _ in ww]
        wy = [w["delta"] for _, w in ww]
        ax_a.errorbar(
            wx,
            wy,
            yerr=_err(wy, [w["ci"] for _, w in ww]),
            fmt="D",
            color=COLORS["delta_ww"],
            label="Δ_ww (within-window rows)",
            capsize=2,
        )
    fa = [(inp.pos[r], per[r]["delta_fa"]) for r in rungs if "delta_fa" in per[r]]
    if fa:
        fx = [x for x, _ in fa]
        fy = [w["delta"] for _, w in fa]
        ax_a.errorbar(
            fx,
            fy,
            yerr=_err(fy, [w["ci"] for _, w in fa]),
            fmt="v",
            color=COLORS["delta_fa"],
            label="Δ_FA (full-attention layer)",
            capsize=2,
        )
    band_lo, band_hi, band_x = [], [], []
    for r in rungs:
        rob = per[r].get("robustness_matched_n") or {}
        d6 = (rob.get("delta_6k_matched") or {}).get("delta")
        arms = [v["delta_vs_companion"] for k, v in rob.items() if k.startswith("gen4_")]
        if d6 is not None and arms:
            band_x.append(inp.pos[r])
            band_lo.append(d6 + min(arms))
            band_hi.append(d6 + max(arms))
    if band_x:
        ax_a.fill_between(
            band_x,
            band_lo,
            band_hi,
            color="0.8",
            alpha=0.5,
            label="order/set robustness (matched 6k)",
        )
    ceil = [((per[r].get("ceiling_delta_paired") or {}).get("spearman_brown")) for r in rungs]
    if any(v is not None and np.isfinite(v) for v in np.asarray(ceil, dtype=float)):
        ax_a.plot(
            xs,
            np.asarray(ceil, dtype=float),
            "^:",
            color=COLORS["ceiling"],
            label="paired Δ ceiling (SB)",
        )
    ax_a.axhline(0.0, color="0.6", linewidth=0.7)
    _apply_rung_axis(ax_a, rungs, inp.pos, [inp.ticklabels[inp.rungs.index(r)] for r in rungs])
    ax_a.set_ylabel("Δ R² at layer*")
    ax_a.set_title("Δ(T) pooled at sliding layer* — over-window rows included")
    ax_a.legend(fontsize=6.5)

    cmap = plt.get_cmap("viridis")
    dose_any = False
    for k, r in enumerate(rungs):
        dose = per[r].get("dose_panel_r2_at_star") or {}
        ks = sorted(int(x) for x in dose)
        if len(ks) < 2:
            continue
        dose_any = True
        ax_b.plot(
            ks, [dose[str(x)] for x in ks], "o-", color=cmap(k / max(len(rungs) - 1, 1)), label=r
        )
    if dose_any:
        ax_b.set_xticks([0, 1, 4, 16])
        ax_b.set_xlabel("k exemplars")
        ax_b.set_ylabel("R² at layer*")
        ax_b.set_title("dose panel")
        ax_b.legend(fontsize=6.5, title="rung")
    ow = inp.kshot.get("over_window_fracs") or {}
    ow_rungs = [r for r in rungs if f"{r}_k4" in ow]
    if ow_rungs:
        fr = [ow[f"{r}_k4"]["frac_over_window"] for r in ow_rungs]
        ax_c.bar(range(len(ow_rungs)), fr, color=COLORS["delta"], alpha=0.8)
        ax_c.set_xticks(range(len(ow_rungs)))
        ax_c.set_xticklabels(ow_rungs, fontsize=7)
        ax_c.set_ylabel("fraction of rows over window")
        ax_c.set_title("over-window fraction (k=4)")
    if ax_s is not None:
        sx = [inp.pos[r] for r in strip_rungs]
        ax_s.plot(sx, [trunc[r]["gen0"] for r in strip_rungs], "o-", color="0.35", label="gen0 arm")
        ax_s.plot(
            sx,
            [trunc[r]["gen4"] for r in strip_rungs],
            "s--",
            color=COLORS["delta"],
            label="gen4 arm",
        )
        ax_s.set_ylim(0, 1.02)
        ax_s.set_ylabel("truncation rate")
        ax_s.set_title("per-rung per-arm truncation-censoring rate (reported, never filtered)")
        _apply_rung_axis(ax_s, rungs, inp.pos, [inp.ticklabels[inp.rungs.index(r)] for r in rungs])
        ax_s.legend(fontsize=7)
    savefig_paper(fig, "fig4_kshot_curve", dir=out_dir)
    plt.close(fig)
    return "fig4_kshot_curve"


def fig4b_kshot_perfold(inp: Inputs, out_dir: Path) -> str | None:
    """Companion: per-fold Δ (diag4 − diag0 at layer*), fold-labeled."""
    fig, ax = plt.subplots(figsize=(8.5, 4.0), layout="constrained")
    plotted = False
    for r in inp.rungs:
        f0 = inp.fold_star_r2(r, "diag0")
        f4 = inp.fold_star_r2(r, "cell_diag4")
        shared = sorted(set(f0) & set(f4))
        if not shared:
            continue
        plotted = True
        xs = [inp.pos[r]] * len(shared)
        ys = [f4[f] - f0[f] for f in shared]
        ax.scatter(xs, ys, s=14, color=COLORS["delta"], alpha=0.6)
        _label_points(ax, xs, ys, [f"f{f}" for f in shared], COLORS["delta"])
    if not plotted:
        print("[figures] skip fig4b: no paired diag0/diag4 percell shards", flush=True)
        plt.close(fig)
        return None
    ax.axhline(0.0, color="0.6", linewidth=0.7)
    _apply_rung_axis(ax, inp.rungs, inp.pos, inp.ticklabels)
    ax.set_ylabel("per-fold Δ R² at layer*")
    ax.set_title("per-fold Δ (4-shot − 0-shot)")
    savefig_paper(fig, "fig4b_kshot_perfold", dir=out_dir)
    plt.close(fig)
    return "fig4b_kshot_perfold"


def fig5a_per_class(inp: Inputs, out_dir: Path) -> str | None:
    classes = sorted({c for r in inp.rungs for c in (inp.per_rung[r].get("per_class") or {})})
    if not classes:
        print("[figures] skip fig5a: no per-class reads", flush=True)
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    pal = paper_palette(max(len(classes), 3))
    for k, cls in enumerate(classes):
        pts = [
            (inp.pos[r], (inp.per_rung[r]["per_class"].get(cls) or {}).get("r2"))
            for r in inp.rungs
            if cls in (inp.per_rung[r].get("per_class") or {})
        ]
        px = [x for x, y in pts if y is not None]
        py = [y for _, y in pts if y is not None]
        if px:
            ax.plot(px, py, "o-", color=pal[k], label=cls, alpha=0.85)
    _apply_rung_axis(ax, inp.rungs, inp.pos, inp.ticklabels)
    ax.set_ylabel("per-class pooled OOF R² at layer*")
    ax.set_title("per-class formation curves D_c(T)")
    ax.legend(fontsize=6.5)
    savefig_paper(fig, "fig5a_per_class", dir=out_dir)
    plt.close(fig)
    return "fig5a_per_class"


def fig5b_layer_sweep(inp: Inputs, out_dir: Path) -> str | None:
    r2d = np.asarray(inp.nullmat["r2_draws"])  # (n_rungs, n_draws, n_layers)
    rungs = [str(x) for x in inp.nullmat["rungs"]]
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    cmap = plt.get_cmap("viridis")
    for k, r in enumerate(rungs):
        ax.plot(
            inp.layers17,
            np.nanmean(r2d[k], axis=0),
            "o-",
            color=cmap(k / max(len(rungs) - 1, 1)),
            label=r,
            alpha=0.85,
            markersize=3,
        )
    fa = [x for x in inp.layers17 if x in C2.OLMO3_FULL_ATTENTION_LAYERS]
    for x in fa:
        ax.axvline(x, color="0.8", linestyle=":", linewidth=1)
    ax.axvline(inp.layer_star, color="black", linewidth=1, label=f"layer* = {inp.layer_star}")
    ax.set_xlabel("layer (dotted grey = full-attention layers)")
    ax.set_ylabel("mean bootstrap R²")
    ax.set_title("17-layer diagonal sweep per rung")
    ax.legend(fontsize=6, ncols=2, title="rung")
    savefig_paper(fig, "fig5b_layer_sweep", dir=out_dir)
    plt.close(fig)
    return "fig5b_layer_sweep"


def fig5c_best_layer(inp: Inputs, out_dir: Path) -> str | None:
    best = np.asarray(inp.nullmat["best_layer_idx"])  # (n_rungs, n_draws)
    rungs = [str(x) for x in inp.nullmat["rungs"]]
    n_layers = len(inp.layers17)
    mass = np.stack(
        [np.bincount(best[k], minlength=n_layers) / best.shape[1] for k in range(len(rungs))]
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    im = ax.imshow(mass, aspect="auto", cmap="magma", origin="lower")
    ax.set_yticks(range(len(rungs)))
    ax.set_yticklabels(rungs, fontsize=7)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(inp.layers17, fontsize=6)
    ax.set_xlabel("layer")
    ax.set_title("per-draw best-layer selection mass (same-selection read)")
    fig.colorbar(im, ax=ax, label="draw fraction")
    savefig_paper(fig, "fig5c_best_layer_selection", dir=out_dir)
    plt.close(fig)
    return "fig5c_best_layer_selection"


def fig5d_knn(inp: Inputs, out_dir: Path) -> str | None:
    fig, ax = plt.subplots(figsize=(7.5, 4.0), layout="constrained")
    plotted = False
    for metric, fmt in (("euclidean", "o-"), ("cosine", "s--")):
        ys = [(inp.per_rung[r].get("knn_acc_at_1") or {}).get(metric) for r in inp.rungs]
        if any(v is not None for v in ys):
            plotted = True
            ax.plot(
                [inp.pos[r] for r in inp.rungs],
                np.asarray(ys, dtype=float),
                fmt,
                color=COLORS["diag"] if metric == "euclidean" else COLORS["dtilde"],
                label=f"kNN acc@1 ({metric})",
            )
    if not plotted:
        print("[figures] skip fig5d: no kNN reads", flush=True)
        plt.close(fig)
        return None
    _apply_rung_axis(ax, inp.rungs, inp.pos, inp.ticklabels)
    ax.set_ylabel("retrieval acc@1 (held-out pool)")
    ax.set_title("kNN retrieval of the true target")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig5d_knn_retrieval", dir=out_dir)
    plt.close(fig)
    return "fig5d_knn_retrieval"


def fig5e_lambda_dof(inp: Inputs, out_dir: Path) -> str | None:
    fig, (ax_l, ax_d) = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    plotted = False
    for r in inp.rungs:
        lam = [v for v in inp.per_rung[r].get("lambda_star_by_fold", []) if v is not None]
        dof = [v for v in inp.per_rung[r].get("dof_by_fold", []) if v is not None]
        if lam:
            plotted = True
            ax_l.scatter([inp.pos[r]] * len(lam), lam, s=12, color=COLORS["diag"], alpha=0.6)
        if dof:
            ax_d.scatter([inp.pos[r]] * len(dof), dof, s=12, color=COLORS["dtilde"], alpha=0.6)
    if not plotted:
        print("[figures] skip fig5e: no selector records", flush=True)
        plt.close(fig)
        return None
    for ax, ylab, title in (
        (ax_l, "λ* per fold", "selected ridge λ*"),
        (ax_d, "dof per fold", "effective dof (cap 0.9·n)"),
    ):
        _apply_rung_axis(ax, inp.rungs, inp.pos, inp.ticklabels)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    ax_l.set_yscale("log")
    savefig_paper(fig, "fig5e_lambda_dof_selector", dir=out_dir)
    plt.close(fig)
    return "fig5e_lambda_dof_selector"


def fig5f_overwindow(inp: Inputs, out_dir: Path) -> str | None:
    ow = inp.kshot.get("over_window_fracs") or {}
    if not ow:
        print("[figures] skip fig5f: no over-window diagnostics", flush=True)
        return None
    keys = sorted(ow)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.0), layout="constrained")
    ax_a.bar(
        range(len(keys)),
        [ow[k]["frac_over_window"] for k in keys],
        color=COLORS["delta"],
        alpha=0.8,
    )
    ax_a.set_xticks(range(len(keys)))
    ax_a.set_xticklabels(keys, rotation=60, fontsize=6)
    ax_a.set_ylabel("fraction over window")
    ax_a.set_title(f"over-window fraction per (rung, k), window={inp.kshot.get('window')}")
    ax_b.plot(
        range(len(keys)),
        [ow[k]["len_p50"] for k in keys],
        "o-",
        color=COLORS["diag"],
        label="p50 prompt+answer tokens",
    )
    ax_b.plot(
        range(len(keys)),
        [ow[k]["len_p90"] for k in keys],
        "s--",
        color=COLORS["dtilde"],
        label="p90",
    )
    ax_b.axhline(
        float(inp.kshot.get("window") or 4096), color="0.4", linestyle=":", label="sliding window"
    )
    ax_b.set_xticks(range(len(keys)))
    ax_b.set_xticklabels(keys, rotation=60, fontsize=6)
    ax_b.set_ylabel("tokens")
    ax_b.set_title("sequence-length percentiles")
    ax_b.legend(fontsize=7)
    savefig_paper(fig, "fig5f_over_window_lengths", dir=out_dir)
    plt.close(fig)
    return "fig5f_over_window_lengths"


def fig5g_trained_only(inp: Inputs, out_dir: Path) -> str | None:
    tos = inp.diag.get("trained_only_sensitivity") or {}
    per = (tos.get("trained_only_intersection") or {}).get("per_rung") or {}
    if not per:
        print(
            f"[figures] skip fig5g: trained-only read unavailable ({tos.get('status')})", flush=True
        )
        return None
    rungs = [r for r in inp.rungs if r in per]
    xs = [inp.pos[r] for r in rungs]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    ax.plot(
        xs,
        [inp.per_rung[r]["r2_star"] for r in rungs],
        "o-",
        color=COLORS["diag"],
        label="headline (shared intersection)",
    )
    ax.plot(
        xs,
        [per[r]["r2"] for r in rungs],
        "s--",
        color=COLORS["dtilde"],
        label="trained-rungs-only intersection",
    )
    ax.plot(
        xs,
        [per[r]["identity_r2"] for r in rungs],
        ":",
        color=COLORS["identity"],
        label="identity+bias (trained-only rows)",
    )
    dm = tos.get("d_main_full_unflagged")
    if dm and "main" in inp.pos:
        ax.plot(
            inp.pos["main"],
            dm["r2"],
            "*",
            markersize=11,
            color="black",
            label="D(main), full unflagged rows",
        )
    _apply_rung_axis(ax, rungs, inp.pos, [inp.ticklabels[inp.rungs.index(r)] for r in rungs])
    ax.set_ylabel("pooled OOF R² at layer*")
    ax.set_title("intersection-denominator sensitivity")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig5g_trained_only_sensitivity", dir=out_dir)
    plt.close(fig)
    return "fig5g_trained_only_sensitivity"


def fig5h_natgen(inp: Inputs, out_dir: Path) -> str | None:
    per = inp.kshot.get("per_rung") or {}
    rows = [
        (r, per[r]["robustness_matched_n"]["natgen_vs_plain"])
        for r in inp.rungs
        if "natgen_vs_plain" in (per.get(r, {}).get("robustness_matched_n") or {})
    ]
    if not rows:
        print("[figures] skip fig5h: no native-gen robustness cells", flush=True)
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.0), layout="constrained")
    w = 0.35
    xs = np.arange(len(rows))
    ax.bar(
        xs - w / 2,
        [v["native_r2"] for _, v in rows],
        w,
        color=COLORS["diag"],
        label="native chat render",
    )
    ax.bar(
        xs + w / 2,
        [v["plain_r2_matched"] for _, v in rows],
        w,
        color=COLORS["identity"],
        label="plain render (matched 2k rows)",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([r for r, _ in rows])
    ax.set_ylabel("pooled OOF R² at layer*")
    ax.set_title("native-gen vs plain-render (matched rows)")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig5h_natgen_vs_plain", dir=out_dir)
    plt.close(fig)
    return "fig5h_natgen_vs_plain"


def fig5i_robust_arms(inp: Inputs, out_dir: Path) -> str | None:
    per = inp.kshot.get("per_rung") or {}
    arms = ("gen4_o2", "gen4_o3", "gen4_s2", "gen4_s3")
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for r in inp.rungs:
        rob = per.get(r, {}).get("robustness_matched_n") or {}
        for a in arms:
            if a in rob:
                rows.append((r, a, rob[a]))
    if not rows:
        print("[figures] skip fig5i: no order/set robustness arms", flush=True)
        return None
    fig, ax = plt.subplots(figsize=(max(6.5, 0.5 * len(rows) + 2), 4.0), layout="constrained")
    ys = [v["delta_vs_companion"] for _, _, v in rows]
    ax.errorbar(
        range(len(rows)),
        ys,
        yerr=_err(ys, [v["delta_ci"] for _, _, v in rows]),
        fmt="o",
        color=COLORS["delta_ww"],
        capsize=2,
    )
    ax.axhline(0.0, color="0.6", linewidth=0.7)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r}\n{a[5:]}" for r, a, _ in rows], fontsize=6.5)
    ax.set_ylabel("R²(arm) − R²(O1 companion), matched 6k")
    ax.set_title("order/set exemplar robustness (subset-vs-subset)")
    savefig_paper(fig, "fig5i_robustness_arms", dir=out_dir)
    plt.close(fig)
    return "fig5i_robustness_arms"


def fig5j_ceilings(inp: Inputs, out_dir: Path) -> str | None:
    per_k = inp.kshot.get("per_rung") or {}
    fig, ax = plt.subplots(figsize=(8.0, 4.0), layout="constrained")
    plotted = False
    series = (
        ("ceiling_0shot", inp.per_rung, "0-shot ceiling", "o-", COLORS["diag"]),
        ("ceiling_4shot", per_k, "4-shot ceiling", "s--", COLORS["delta"]),
        ("ceiling_delta_paired", per_k, "paired Δ ceiling", "D:", COLORS["delta_ww"]),
    )
    for key, src, label, fmt, color in series:
        pts = [
            (inp.pos[r], (src.get(r, {}).get(key) or {}).get("spearman_brown")) for r in inp.rungs
        ]
        px = [x for x, y in pts if y is not None and np.isfinite(y)]
        py = [y for _, y in pts if y is not None and np.isfinite(y)]
        if px:
            plotted = True
            ax.plot(px, py, fmt, color=color, label=label)
    if not plotted:
        print("[figures] skip fig5j: no reliability ceilings", flush=True)
        plt.close(fig)
        return None
    _apply_rung_axis(ax, inp.rungs, inp.pos, inp.ticklabels)
    ax.set_ylabel("Spearman–Brown split-half reliability")
    ax.set_title("repeat-generation ceilings per rung")
    ax.legend(fontsize=7)
    savefig_paper(fig, "fig5j_ceilings", dir=out_dir)
    plt.close(fig)
    return "fig5j_ceilings"


def fig5k_censoring(inp: Inputs, out_dir: Path) -> str | None:
    """Exploratory (v8 §6 censoring family): D_nt(T) natural-termination
    overlay vs the headline D(T); the within-rung truncated-vs-natural split;
    Δ_nt/Δ_tt common-status reads vs the pooled Δ."""
    cen = inp.censor
    if not cen or "d_nt" not in cen:
        print("[figures] skip fig5k: no censoring_sensitivity block", flush=True)
        return None
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.5, 4.2), layout="constrained")
    rungs = inp.rungs
    xs = [inp.pos[r] for r in rungs]

    # (a) D(T) headline vs D_nt(T) on the natural-termination subset
    ax_a.plot(
        xs,
        [inp.per_rung[r]["r2_star"] for r in rungs],
        "o-",
        color=COLORS["diag"],
        label="D(T) headline",
    )
    dn = [(inp.pos[r], cen["d_nt"][r]) for r in rungs if "r2" in (cen["d_nt"].get(r) or {})]
    if dn:
        dny = [v["r2"] for _, v in dn]
        ax_a.errorbar(
            [x for x, _ in dn],
            dny,
            yerr=_err(dny, [v.get("ci") for _, v in dn]),
            fmt="s--",
            color=COLORS["delta_ww"],
            label="D_nt(T) natural-termination rows",
            capsize=2,
        )
    _apply_rung_axis(ax_a, rungs, inp.pos, inp.ticklabels)
    ax_a.set_ylabel("pooled OOF R² at layer*")
    ax_a.set_title("D(T) vs D_nt(T)\n(no-natural-row rungs omitted)")
    ax_a.legend(fontsize=7)

    # (b) within-rung truncated-vs-natural split (where reported)
    split = cen.get("stratified_split") or {}
    plotted_b = False
    for label, key, color, fmt in (
        ("natural rows", "natural", COLORS["delta_ww"], "s--"),
        ("truncated rows", "truncated", "0.35", "o-"),
    ):
        pts = [
            (inp.pos[r], split[r][key]["r2"])
            for r in rungs
            if "r2" in (split.get(r, {}).get(key) or {})
        ]
        if pts:
            plotted_b = True
            ax_b.plot([x for x, _ in pts], [y for _, y in pts], fmt, color=color, label=label)
    if plotted_b:
        _apply_rung_axis(ax_b, rungs, inp.pos, inp.ticklabels)
        ax_b.set_ylabel("pooled OOF R² at layer*")
        ax_b.set_title(f"truncated vs natural strata (floor n≥{cen.get('split_floor')})")
        ax_b.legend(fontsize=7)
    else:
        ax_b.set_title("stratified split: no rung clears the floor")

    # (c) Δ(T) pooled vs the Δ_nt/Δ_tt common-status reads
    per = inp.kshot.get("per_rung") or {}
    krungs = [r for r in rungs if r in per]
    if krungs:
        kx = [inp.pos[r] for r in krungs]
        ax_c.plot(
            kx, [per[r]["delta"] for r in krungs], "o-", color=COLORS["delta"], label="Δ pooled"
        )
        for label, color, fmt in (
            ("delta_nt", COLORS["delta_ww"], "s--"),
            ("delta_tt", "0.35", "v:"),
        ):
            pts = [
                (inp.pos[r], per[r][label]["delta"])
                for r in krungs
                if "delta" in (per[r].get(label) or {})
            ]
            if pts:
                ax_c.plot(
                    [x for x, _ in pts],
                    [y for _, y in pts],
                    fmt,
                    color=color,
                    label={
                        "delta_nt": "Δ_nt (both arms natural)",
                        "delta_tt": "Δ_tt (both arms truncated)",
                    }[label],
                )
        ax_c.axhline(0.0, color="0.6", linewidth=0.7)
        _apply_rung_axis(
            ax_c, krungs, inp.pos, [inp.ticklabels[inp.rungs.index(r)] for r in krungs]
        )
        ax_c.set_ylabel("Δ R² at layer*")
        ax_c.set_title("Δ vs common-status Δ_nt/Δ_tt")
        ax_c.legend(fontsize=7)
    savefig_paper(fig, "fig5k_censoring", dir=out_dir)
    plt.close(fig)
    return "fig5k_censoring"


FIGURES = (
    fig1_formation,
    fig1b_formation_perfold,
    fig2_decomposition,
    fig2b_decomposition_band,
    fig3_retention,
    fig3b_retention_perpair,
    fig4_kshot,
    fig4b_kshot_perfold,
    fig5a_per_class,
    fig5b_layer_sweep,
    fig5c_best_layer,
    fig5d_knn,
    fig5e_lambda_dof,
    fig5f_overwindow,
    fig5g_trained_only,
    fig5h_natgen,
    fig5i_robust_arms,
    fig5j_ceilings,
    fig5k_censoring,
)


# ── main ─────────────────────────────────────────────────────────────────────


def _default_eval_dir() -> Path:
    import os

    env = os.environ.get("EPM_ISSUE2544_EVAL_DIR")
    return Path(env) if env else PROJECT_ROOT / "eval_results" / "issue_2544"


def main() -> None:
    ap = argparse.ArgumentParser(description="#2544 P5 figure driver")
    ap.add_argument("--eval-dir", default=None, help="eval_results/issue_2544 root")
    ap.add_argument("--out-dir", default=None, help="figures output dir")
    ap.add_argument("--smoke", action="store_true", help="smoke tree: divert output off git")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-figures", action="store_true")
    args = ap.parse_args()

    if args.list_figures:
        print(" ".join(fn.__name__ for fn in FIGURES))
        sys.exit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)

    eval_dir = Path(args.eval_dir) if args.eval_dir else _default_eval_dir()
    committed = PROJECT_ROOT / "figures" / "issue_2544"
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.smoke:
        out_dir = eval_dir.parents[1] / "figures" / "issue_2544"
    else:
        out_dir = committed
    if args.smoke and out_dir.resolve() == committed.resolve():
        raise SystemExit(
            "--smoke refuses the committed figures dir — smoke outputs never "
            "overwrite committed artifacts (pass --out-dir off the repo tree)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    inp = Inputs(eval_dir)
    print(
        f"[figures] eval_dir={eval_dir} out_dir={out_dir} rungs={inp.rungs} "
        f"layer_star={inp.layer_star} smoke={inp.diag.get('smoke')}",
        flush=True,
    )
    saved: list[str] = []
    for fn in FIGURES:
        stem = fn(inp, out_dir)
        if stem:
            saved.append(stem)
            print(f"[figures] saved {stem}", flush=True)
    heroes = {"fig1_formation_curve", "fig2_decomposition", "fig3_retention", "fig4_kshot_curve"}
    missing = heroes - set(saved)
    if missing:
        raise RuntimeError(f"hero figure(s) not produced: {sorted(missing)} — inputs incomplete")
    print(f"[figures] done: {len(saved)} figures -> {out_dir}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
