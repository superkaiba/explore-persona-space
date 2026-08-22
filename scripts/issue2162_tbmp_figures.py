#!/usr/bin/env python3
"""Issue #2162 tbmp — plan §6 figures (turn-boundary multipatch round).

Renders the six registered figures + the exploratory dump from the round's
VM-side tables (no torch, no judge calls — same light-import contract as the
parent ``issue2162_figures.py``, whose helpers this script reuses):

- ``tb_hero``        per base x depth: joint-tb F per arm (registered space),
                     pair-clustered 95% CIs, per-pair points, parent single-ce
                     steered read overlaid (parent_ref re-aggregation, §12.8).
- ``tb_sweep``       per-boundary steered - shuffled ΔF vs boundary index k,
                     per base x depth; parent ce read as the final-k point
                     (provenance-labeled; persona parent points are the
                     target-only re-aggregation, never netted-in-target panel).
- ``tb_rawscale``    raw-scale movement per arm x depth, joint tb vs parent
                     single-ce; ``all`` + ``surviving`` panels.
- ``tb_identity_d1`` per-pair scatter tb@d1 F vs parent ce F (steered +
                     shuffled, NETTED space — the deliberate §6 exception:
                     G2 runs in the parent's netted space), ±0.10 gate band,
                     surviving-pool annotation.
- ``tb_control``     control cell joint tb per arm (edit-artifact read).
- ``tb_target_only`` persona cells: target-only vs netted per arm x depth.
- exploratory dump: per-cell per-pair strips, margin-vs-F scatter (when the
  margin leg has landed), cap-hit/coherence tables -> ``captions.json``.

Inputs: ``eval_results/issue_2162/turn_boundary/{f_cells_tb,
null_shuffled_cells_tb, null_crosstype_cells_tb, parent_ref_cells_tb}.jsonl``
+ ``stats_tb.json`` + ``identity_gate.json`` [+ ``margin_cells_tb.jsonl``],
plus the parent's committed ``f_metrics`` tables (identity panel only).
Outputs: ``figures/issue_2162/tb_*.png`` (+ .pdf/.meta.json) and
``eval_results/issue_2162/turn_boundary/captions.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps bind BEFORE the heavy imports (#847)

import issue2162_figures as F  # noqa: E402  (light: matplotlib/numpy only)
import matplotlib.pyplot as plt  # noqa: E402  (Agg backend set by F's import)
import numpy as np  # noqa: E402

logger = logging.getLogger("issue2162.tbmp.figures")

# Local constants pinned equal to the driver/analysis modules by
# tests/test_issue2162_tbmp_analysis.py (so this script never imports the
# torch-heavy driver at render time — the parent figures-script pattern).
BASES: tuple[str, ...] = ("instr_format", "persona_prompted")
DEPTHS: tuple[str, ...] = ("d1", "d3", "d5")
CONTROL_CELL = "recency_fact_user_name_d5"
FINAL_K = {
    "recency_instr_format_d3": 3,
    "recency_instr_format_d5": 5,
    "recency_persona_prompted_d3": 3,
    "recency_persona_prompted_d5": 5,
}
BOOT_B = 10_000
BOOT_SEED = 21621

ARM_COLORS = F.ARM_COLORS
PARENT_COLOR = "#222222"
SPACE_COLORS = {"netted": "#8c564b", "target_only": "#2ca02c"}


def depth_cell(base: str, depth: str) -> str:
    return base if depth == "d1" else f"recency_{base}_{depth}"


def registered_space_label(base: str) -> str:
    return "target-only" if base == "persona_prompted" else "netted"


def _boot_ci(vals: list[float], seed: int = BOOT_SEED) -> tuple[float | None, float | None]:
    """Pair-clustered percentile bootstrap CI of the mean (pairs = the
    resampled unit; each value is one pair's statistic)."""
    if len(vals) < 2:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(BOOT_B, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _surviving(rows: list[dict], value_key: str = "f_beh") -> list[dict]:
    return [
        r
        for r in rows
        if r.get(value_key) is not None
        and r.get("separation") is not None
        and abs(r["separation"]) >= F.SEPARATION_BAR
    ]


def _index(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        out[(r["cell"], r["slot"])].append(r)
    return out


def _errbar(ax, x: float, vals: list[float], color: str, label: str | None = None) -> None:
    if not vals:
        return
    m = float(np.mean(vals))
    lo, hi = _boot_ci(vals)
    e_lo, e_hi = F._err(lo, hi, m)
    ax.errorbar(
        [x],
        [m],
        yerr=[[e_lo], [e_hi]],
        fmt="o",
        color=color,
        capsize=3,
        markersize=7,
        zorder=4,
        label=label,
    )


def _points(ax, x: float, vals: list[float], color: str, jitter: float = 0.04) -> None:
    if not vals:
        return
    rng = np.random.default_rng(BOOT_SEED + len(vals))
    xs = x + rng.uniform(-jitter, jitter, size=len(vals))
    ax.scatter(xs, vals, s=9, color=color, alpha=0.30, linewidths=0, zorder=2)


def _require_nonempty_panels(fig, fig_name: str, empty_panels: list[str]) -> None:
    """Fail LOUD on empty panels — the #1112 empty-figure class: a silent skip
    over missing rows must never ship a blank render presented as a result."""
    if empty_panels:
        plt.close(fig)
        raise RuntimeError(
            f"[figures] {fig_name}: {len(empty_panels)} panel(s) rendered EMPTY "
            f"({', '.join(empty_panels)}) — upstream tables carry none of the expected rows"
        )


# ── figures ───────────────────────────────────────────────────────────


def fig_tb_hero(tb: dict[str, dict], ref: dict[str, dict], fig_dir: Path, inputs: list[Path]):
    fig, axes = plt.subplots(1, len(BASES), figsize=(11, 4.4), sharey=True)
    offs = {"steered": -0.22, "shuffled": -0.07, "crosstype": 0.08}
    empty_panels: list[str] = []
    for ax, base in zip(np.atleast_1d(axes), BASES, strict=True):
        n_series = 0
        for d_i, depth in enumerate(DEPTHS):
            cell = depth_cell(base, depth)
            for arm, off in offs.items():
                vals = [r["f_beh"] for r in _surviving(tb[arm]) if r["cell"] == cell]
                n_series += bool(vals)
                _points(ax, d_i + off, vals, ARM_COLORS[arm])
                _errbar(ax, d_i + off, vals, ARM_COLORS[arm], label=arm if d_i == 0 else None)
            ref_vals = [
                r["f_beh"]
                for r in _surviving(ref["steered"])
                if r["cell"] == cell and r["slot"] == "ce"
            ]
            n_series += bool(ref_vals)
            _errbar(
                ax,
                d_i + 0.23,
                ref_vals,
                PARENT_COLOR,
                label="parent single-ce (steered)" if d_i == 0 else None,
            )
        if n_series == 0:
            empty_panels.append(base)
        ax.set_xticks(range(len(DEPTHS)), DEPTHS)
        ax.set_title(f"{base} (registered: {registered_space_label(base)})")
        ax.axhline(0.0, color="#bbbbbb", lw=0.8, zorder=1)
        ax.set_xlabel("depth")
    _require_nonempty_panels(fig, "tb_hero", empty_panels)
    np.atleast_1d(axes)[0].set_ylabel("F (registered space)")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    F._save(fig, fig_dir, "tb_hero", inputs)


def fig_tb_sweep(tb: dict[str, dict], ref: dict[str, dict], fig_dir: Path, inputs: list[Path]):
    # constrained_layout: without it the top row's "boundary index k" xlabel
    # overlaps the bottom row's panel titles and both render garbled (2x2 grid,
    # per-panel titles + xlabels). Layout only — no data or scale change.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True, constrained_layout=True)
    cells = sorted(FINAL_K)
    empty_panels: list[str] = []
    for ax, cell in zip(axes.ravel(), cells, strict=True):
        n_series = 0
        st = {(r["pair_id"], r["slot"]): r for r in _surviving(tb["steered"]) if r["cell"] == cell}
        sh = {(r["pair_id"], r["slot"]): r for r in _surviving(tb["shuffled"]) if r["cell"] == cell}
        for k in range(1, FINAL_K[cell]):
            slot = f"tbk{k}"
            diffs = [
                st[key]["f_beh"] - sh[key]["f_beh"]
                for key in sorted(st)
                if key[1] == slot and key in sh
            ]
            n_series += bool(diffs)
            _points(ax, k, diffs, ARM_COLORS["steered"])
            _errbar(ax, k, diffs, ARM_COLORS["steered"])
        ref_st = {
            r["pair_id"]: r["f_beh"]
            for r in _surviving(ref["steered"])
            if r["cell"] == cell and r["slot"] == "ce"
        }
        ref_sh = {
            r["pair_id"]: r["f_beh"]
            for r in _surviving(ref["shuffled"])
            if r["cell"] == cell and r["slot"] == "ce"
        }
        ref_diffs = [ref_st[p] - ref_sh[p] for p in sorted(ref_st) if p in ref_sh]
        n_series += bool(ref_diffs)
        if n_series == 0:
            empty_panels.append(cell)
        k_final = FINAL_K[cell]
        if ref_diffs:
            m = float(np.mean(ref_diffs))
            lo, hi = _boot_ci(ref_diffs)
            e_lo, e_hi = F._err(lo, hi, m)
            ax.errorbar(
                [k_final],
                [m],
                yerr=[[e_lo], [e_hi]],
                fmt="D",
                color=PARENT_COLOR,
                capsize=3,
                markersize=7,
                zorder=4,
                label=f"parent ce (k={k_final})",
            )
        ax.axhline(0.0, color="#bbbbbb", lw=0.8, zorder=1)
        ax.set_title(cell)
        ax.set_xlabel("boundary index k")
        ax.set_xticks(range(1, k_final + 1))
        ax.legend(fontsize=8, loc="upper left")
    _require_nonempty_panels(fig, "tb_sweep", empty_panels)
    for ax in axes[:, 0]:
        ax.set_ylabel("ΔF steered − shuffled (registered)")
    F._save(fig, fig_dir, "tb_sweep", inputs)


def _rs_index(payload: dict) -> dict[tuple[str, str], dict]:
    return {(r["cell"], r["subset"]): r for r in payload["rows"]}


def fig_tb_rawscale(rawscale: dict, parent_rs: dict, fig_dir: Path, inputs: list[Path]) -> None:
    """Raw-scale (denominator-free, judge-contrast units) per arm x depth —
    sourced from the DECLARED artifacts (plan §9 P5 ``rawscale_tb.json`` + the
    parent's committed ``recency_rawscale.json``), never recomputed from the F
    tables; every bar carries its pair-clustered bootstrap 95% CI (B=10000,
    seed 21620) via ``<arm>_ci95`` where the artifact provides one (the parent
    file predates the null-CI arm and carries steered CIs only)."""
    arms = ("steered", "shuffled", "crosstype")
    tb_idx = _rs_index(rawscale)
    ref_idx = _rs_index(parent_rs)
    fig, axes = plt.subplots(2, len(BASES), figsize=(12, 7.5), sharey="row")
    empty_panels: list[str] = []
    for row_i, pool in enumerate(("all", "surviving")):
        for col_i, base in enumerate(BASES):
            ax = axes[row_i, col_i]
            width = 0.13
            n_bars = 0
            for d_i, depth in enumerate(DEPTHS):
                cell = depth_cell(base, depth)
                for a_i, arm in enumerate(arms):
                    x0 = d_i + (a_i - 1) * 2.2 * width
                    for src_i, (idx, hatch, tag) in enumerate(
                        ((tb_idx, None, "tb"), (ref_idx, "//", "parent ce"))
                    ):
                        row = idx.get((cell, pool))
                        if row is None:
                            # A missing (cell, pool) row from ONE source is a
                            # legitimate coverage gap; a panel with ZERO bars
                            # is the #1112 empty-figure class — see the
                            # _require_nonempty_panels check below.
                            continue
                        n_bars += 1
                        m = row[f"{arm}_mean"]
                        ci = row.get(f"{arm}_ci95")
                        lo, hi = ci if ci else (None, None)
                        e_lo, e_hi = F._err(lo, hi, m)
                        ax.bar(
                            x0 + (src_i - 0.5) * width,
                            m,
                            width,
                            yerr=[[e_lo], [e_hi]],
                            capsize=2,
                            error_kw={"lw": 0.9},
                            color=ARM_COLORS[arm],
                            hatch=hatch,
                            edgecolor="white" if hatch else None,
                            label=(
                                f"{arm} ({tag})" if d_i == 0 and row_i == 0 and col_i == 0 else None
                            ),
                        )
            if n_bars == 0:
                empty_panels.append(f"{base}/{pool}")
            ax.set_xticks(range(len(DEPTHS)), DEPTHS)
            ax.axhline(0.0, color="#bbbbbb", lw=0.8)
            ax.set_title(f"{base} — {pool} pairs")
            if col_i == 0:
                ax.set_ylabel("raw movement (judge-contrast units)")
    _require_nonempty_panels(fig, "tb_rawscale", empty_panels)
    axes[0, 0].legend(fontsize=7, ncol=2)
    F._save(fig, fig_dir, "tb_rawscale", inputs)


def fig_tb_identity_d1(
    tb: dict[str, dict],
    parent_metrics_dir: Path,
    gate: dict,
    fig_dir: Path,
    inputs: list[Path],
):
    parent_files = {"steered": "f_cells.jsonl", "shuffled": "null_shuffled_cells.jsonl"}
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    markers = {"instr_format": "o", "persona_prompted": "s"}
    all_xy: list[float] = []
    for arm, fname in parent_files.items():
        parent_f = {
            (r["pair_id"], r["cell"]): r["f_beh"]
            for r in F._iter_jsonl(parent_metrics_dir / fname)
            if r["cell"] in BASES and r["slot"] == "ce" and r["f_beh"] is not None
        }
        for base in BASES:
            xs, ys = [], []
            for r in _surviving(tb[arm], "f_netted"):
                if r["cell"] != base or r["slot"] != "tb":
                    continue
                x = parent_f.get((r["pair_id"], base))
                if x is None:
                    continue
                xs.append(x)
                ys.append(r["f_netted"])
            ax.scatter(
                xs,
                ys,
                s=22,
                color=ARM_COLORS[arm],
                marker=markers[base],
                alpha=0.75,
                linewidths=0,
                label=f"{arm} / {base}",
            )
            all_xy.extend(xs + ys)
    _require_nonempty_panels(fig, "tb_identity_d1", [] if all_xy else ["tb@d1 vs parent ce"])
    lo, hi = min(all_xy), max(all_xy)
    pad = 0.05 * (hi - lo + 1e-9)
    xs = np.linspace(lo - pad, hi + pad, 50)
    ax.plot(xs, xs, color="#888888", lw=0.9)
    ax.fill_between(xs, xs - 0.10, xs + 0.10, color="#888888", alpha=0.15)
    # Canvas carries the plain descriptor ONLY. The G2 verdict, pool n and
    # per-arm ΔF used to live in this title: it clipped at both figure edges,
    # and per-arm statistics + a gate verdict rendered onto the canvas are the
    # class the standing figure directive keeps OFF it (axes + ticks + legend +
    # titles only). Those facts move to the caption sidecar, which is what the
    # report reads — nothing is lost.
    ax.set_title("tb@d1 vs parent ce (netted)", fontsize=10)
    ax.set_xlabel("parent single-ce F (netted)")
    ax.set_ylabel("tbmp joint-tb@d1 F (netted)")
    ax.legend(fontsize=8)
    F._save(fig, fig_dir, "tb_identity_d1", inputs)


def fig_tb_control(tb: dict[str, dict], stats: dict, fig_dir: Path, inputs: list[Path]):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    n_series = 0
    for a_i, arm in enumerate(("steered", "shuffled", "crosstype")):
        vals = [
            r["f_beh"]
            for r in _surviving(tb[arm])
            if r["cell"] == CONTROL_CELL and r["slot"] == "tb"
        ]
        n_series += bool(vals)
        _points(ax, a_i, vals, ARM_COLORS[arm])
        _errbar(ax, a_i, vals, ARM_COLORS[arm])
    _require_nonempty_panels(fig, "tb_control", [] if n_series else [CONTROL_CELL])
    rec = stats.get("per_cell", {}).get(f"{CONTROL_CELL}|tb", {})
    verdict = (
        "edit-artifact SIGNAL"
        if rec.get("holm_pass") and rec.get("disjoint_vs_tested_nulls")
        else "no edit-artifact"
    )
    ax.set_xticks(range(3), ("steered", "shuffled", "crosstype"))
    ax.axhline(0.0, color="#bbbbbb", lw=0.8)
    ax.set_ylabel("F (netted)")
    ax.set_title(f"control {CONTROL_CELL} joint tb — {verdict}", fontsize=9)
    F._save(fig, fig_dir, "tb_control", inputs)


def fig_tb_target_only(tb: dict[str, dict], fig_dir: Path, inputs: list[Path]):
    arms = ("steered", "shuffled", "crosstype")
    fig, axes = plt.subplots(1, len(arms), figsize=(12, 4.0), sharey=True)
    empty_panels: list[str] = []
    for ax, arm in zip(axes, arms, strict=True):
        n_series = 0
        for d_i, depth in enumerate(DEPTHS):
            cell = depth_cell("persona_prompted", depth)
            rows = [r for r in tb[arm] if r["cell"] == cell and r["slot"] == "tb"]
            for s_i, space in enumerate(("netted", "target_only")):
                key = "f_netted" if space == "netted" else "f_target_only"
                vals = [r[key] for r in _surviving(rows, key)]
                n_series += bool(vals)
                x = d_i + (-0.12 if s_i == 0 else 0.12)
                _points(ax, x, vals, SPACE_COLORS[space])
                _errbar(ax, x, vals, SPACE_COLORS[space], label=space if d_i == 0 else None)
        if n_series == 0:
            empty_panels.append(arm)
        ax.set_xticks(range(len(DEPTHS)), DEPTHS)
        ax.axhline(0.0, color="#bbbbbb", lw=0.8)
        ax.set_title(f"persona cells — {arm}")
        ax.set_xlabel("depth")
    _require_nonempty_panels(fig, "tb_target_only", empty_panels)
    axes[0].set_ylabel("F")
    axes[0].legend(fontsize=8)
    F._save(fig, fig_dir, "tb_target_only", inputs)


def fig_tb_explore_strips(tb: dict[str, dict], fig_dir: Path, inputs: list[Path]):
    units = sorted({(r["cell"], r["slot"]) for rows in tb.values() for r in rows})
    n_cols = 5
    n_rows = int(np.ceil(len(units) / n_cols))
    # constrained_layout: on a multi-row grid each row's panel titles otherwise
    # land on top of the row above's x-tick labels. Layout only.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.6 * n_rows),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for ax in axes.ravel()[len(units) :]:
        ax.axis("off")
    n_series = 0
    for ax, (cell, slot) in zip(axes.ravel(), units, strict=False):
        for a_i, arm in enumerate(("steered", "shuffled", "crosstype")):
            vals = [
                r["f_beh"] for r in _surviving(tb[arm]) if r["cell"] == cell and r["slot"] == slot
            ]
            n_series += bool(vals)
            _points(ax, a_i, vals, ARM_COLORS[arm], jitter=0.12)
            _errbar(ax, a_i, vals, ARM_COLORS[arm])
        ax.set_title(f"{cell}|{slot}", fontsize=7)
        ax.set_xticks(range(3), ("st", "sh", "ct"), fontsize=7)
        ax.axhline(0.0, color="#cccccc", lw=0.6)
    # Exploratory dump: a single (cell, slot) strip may legitimately empty out
    # under the surviving filter — the fail-loud unit is the whole figure.
    _require_nonempty_panels(fig, "tb_explore_strips", [] if n_series else ["all strips"])
    F._save(fig, fig_dir, "tb_explore_strips", inputs)


def fig_tb_margin_scatter(
    tb: dict[str, dict], margin_path: Path, fig_dir: Path, inputs: list[Path]
) -> bool:
    if not margin_path.exists():
        logger.warning("[figures] %s absent (margin leg deferred) — skipping scatter", margin_path)
        return False
    margins = {
        (r["pair_id"], r["slot"], r["arm"]): r.get("margin_shift", r.get("margin_patched"))
        for r in F._iter_jsonl(margin_path)
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    n_series = 0
    for arm in ("steered", "shuffled", "crosstype"):
        xs, ys = [], []
        for r in _surviving(tb[arm]):
            m = margins.get((r["pair_id"], r["slot"], arm))
            if m is None:
                continue
            xs.append(m)
            ys.append(r["f_beh"])
        n_series += bool(xs)
        ax.scatter(xs, ys, s=14, color=ARM_COLORS[arm], alpha=0.55, linewidths=0, label=arm)
    # The absent-file deferred-leg skip above stays; a PRESENT margin file
    # whose rows join to nothing is the silent-empty class — fail loud.
    _require_nonempty_panels(fig, "tb_margin_scatter", [] if n_series else ["margin vs F"])
    ax.set_xlabel("TF margin (shift vs parent floor; patched when floor absent)")
    ax.set_ylabel("F (registered space)")
    ax.set_title("tbmp: margin vs judged F, per (pair x slot x arm)", fontsize=9)
    ax.legend(fontsize=8)
    F._save(fig, fig_dir, "tb_margin_scatter", inputs)
    return True


CAPTIONS = {
    "tb_hero": (
        "Joint turn-boundary (tb) patching F per arm (registered space: netted for "
        "instr_format, target-descriptor-only for persona_prompted), per base x depth; "
        "points = surviving pairs (|separation| >= 0.5), errorbars = pair-clustered "
        "bootstrap 95% CIs (B=10k). Black = the parent's single context-end (ce) steered "
        "read, re-aggregated per §12.8 (persona cells from the parent's per-descriptor "
        "judge scores, never netted-into-target)."
    ),
    "tb_sweep": (
        "Per-boundary single-position sweep: steered - shuffled ΔF (registered space) at "
        "boundary index k, per sweep cell; paired over surviving pairs, pair-clustered "
        "bootstrap 95% CIs. Diamond = the parent's single-ce read plotted at the final "
        "boundary k=n_d (provenance: parent run, re-aggregated per §12.8)."
    ),
    "tb_rawscale": (
        "Raw-scale movement (denominator-free, judge-contrast units: direction x "
        "(patched - floor)) per arm x depth, sourced from rawscale_tb.json + the parent's "
        "committed recency_rawscale.json: solid = joint tb (this round), hatched = parent "
        "single-ce; errorbars = pair-clustered bootstrap 95% CIs (B=10k, seed 21620; the "
        "parent file carries steered CIs only); top = all scored pairs, bottom = "
        "surviving pairs only."
    ),
    "tb_identity_d1": (
        "G2 identity read: per-pair joint-tb@d1 F vs parent single-ce F, NETTED space for "
        "both axes (the deliberate exception to the registered-space convention — G2 runs "
        "in the parent's netted space), steered + shuffled arms over the surviving d1 "
        "pool; band = the ±0.10 gate bar around y=x."
    ),
    "tb_control": (
        "Control cell (recency_fact_user_name_d5) joint-tb F per arm, netted space, "
        "surviving pairs + pair-clustered 95% CIs — the edit-artifact overlay read "
        "(Holm-pass AND CI-disjoint vs tested nulls downgrades the verdict lattice)."
    ),
    "tb_target_only": (
        "Persona cells scored in BOTH spaces per arm x depth: netted (brown) vs "
        "target-descriptor-only (green; the registered space for persona cells), "
        "surviving pairs + pair-clustered 95% CIs."
    ),
    "tb_explore_strips": (
        "Exploratory: per-(cell x slot) strips of per-pair F (registered space) by arm."
    ),
    "tb_margin_scatter": (
        "Exploratory: teacher-forced margin (secondary continuous DV) vs judged F per "
        "(pair x slot x arm)."
    ),
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="Issue #2162 tbmp plan-§6 figures.")
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_2162/turn_boundary"))
    ap.add_argument(
        "--parent-metrics-dir", type=Path, default=Path("eval_results/issue_2162/f_metrics")
    )
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_2162"))
    args = ap.parse_args(argv)

    tb_paths = {
        "steered": args.out_dir / "f_cells_tb.jsonl",
        "shuffled": args.out_dir / "null_shuffled_cells_tb.jsonl",
        "crosstype": args.out_dir / "null_crosstype_cells_tb.jsonl",
    }
    ref_path = args.out_dir / "parent_ref_cells_tb.jsonl"
    stats_path = args.out_dir / "stats_tb.json"
    gate_path = args.out_dir / "identity_gate.json"
    rawscale_path = args.out_dir / "rawscale_tb.json"
    parent_rs_path = args.parent_metrics_dir / "recency_rawscale.json"
    for p in [*tb_paths.values(), ref_path, stats_path, gate_path, rawscale_path, parent_rs_path]:
        assert p.exists(), f"{p} missing — run issue2162_tbmp_analysis.py first"

    tb = {arm: list(F._iter_jsonl(p)) for arm, p in tb_paths.items()}
    ref_rows = list(F._iter_jsonl(ref_path))
    ref = {
        arm: [r for r in ref_rows if r["arm"] == arm]
        for arm in ("steered", "shuffled", "crosstype")
    }
    stats = json.loads(stats_path.read_text())
    gate = json.loads(gate_path.read_text())
    rawscale = json.loads(rawscale_path.read_text())
    parent_rs = json.loads(parent_rs_path.read_text())

    inputs = [*tb_paths.values(), ref_path, stats_path, gate_path]
    fig_tb_hero(tb, ref, args.fig_dir, inputs)
    fig_tb_sweep(tb, ref, args.fig_dir, inputs)
    fig_tb_rawscale(rawscale, parent_rs, args.fig_dir, [rawscale_path, parent_rs_path])
    fig_tb_identity_d1(tb, args.parent_metrics_dir, gate, args.fig_dir, inputs)
    fig_tb_control(tb, stats, args.fig_dir, inputs)
    fig_tb_target_only(tb, args.fig_dir, inputs)
    fig_tb_explore_strips(tb, args.fig_dir, inputs)
    margin_path = args.out_dir / "margin_cells_tb.jsonl"
    margin_done = fig_tb_margin_scatter(tb, margin_path, args.fig_dir, [*inputs, margin_path])

    captions = dict(CAPTIONS)
    if not margin_done:
        captions["tb_margin_scatter"] += " [NOT RENDERED — margin leg deferred]"

    # G2 verdict / pool n / per-arm ΔF live HERE rather than on the canvas
    # (see fig_tb_identity_d1) — the report reads captions, so the facts stay
    # reportable while the plot stays axes+legend+title only.
    # Every field is OPTIONAL by construction: this caption is descriptive, so a
    # gate payload missing a CI or the bar must degrade the sentence, never crash
    # the render (the figures smoke drives a minimal synthetic gate dict).
    def _g2_arm_bit(arm: str, rec: dict) -> str:
        bit = f"{arm} mean ΔF={rec['mean_delta_f']:+.3f}"
        ci = rec.get("ci95")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            bit += f" (95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])"
        return bit

    g2_arms = ", ".join(
        _g2_arm_bit(a, rec)
        for a, rec in sorted(gate.get("per_arm", {}).items())
        if isinstance(rec, dict) and rec.get("mean_delta_f") is not None
    )
    bar = gate.get("bar")
    bar_clause = f" against the |mean ΔF| ≤ {bar} bar" if bar is not None else ""
    captions["tb_identity_d1"] += (
        f" Gate outcome: G2 {'PASS' if gate.get('passed') else 'FAIL'} on a surviving pool of "
        f"n={gate.get('n_surviving_pool')} pairs{bar_clause}"
        + (f"; {g2_arms}." if g2_arms else ".")
    )
    per_cell = stats.get("per_cell", {})
    payload = {
        "captions": captions,
        "tables": {
            "coherent_fraction": {k: r.get("coherent_fraction") for k, r in per_cell.items()},
            "cap_hit_fraction": {k: r.get("cap_hit_fraction") for k, r in per_cell.items()},
        },
    }
    cap_path = args.out_dir / "captions.json"
    cap_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("[figures] wrote %s", cap_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
