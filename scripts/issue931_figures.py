"""Issue #931 figures: hero panels + exploratory dump from eval_results JSONs.

Hero 1 — 28-layer held-out R^2 curves per regime with group-blocked null bands
(frozen layers marked). Hero 2 — cross-regime transfer heatmap (recentered
matched-power PRIMARY fractions at the headline layer, per-cell source /
denominator n annotations). Exploratory: per-novel R^2 scatter (points
labeled), paired correct-vs-swap per group, span-mean vs single-position
within-R^2 comparison, matched power-curve overlay, full-n vs matched-power
fraction comparison, strict-vs-recentered comparison.

Pure JSON -> PNG/PDF (no torch). --fig-dir overrides the output root so smoke
runs never touch the committed figures/ tree.

CLI:
  uv run python scripts/issue931_figures.py [--results-dir eval_results/issue_931]
      [--fig-dir figures/issue_931]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common  # noqa: E402

SCRIPT = "scripts/issue931_figures.py"

REGIME_CELLS = [
    ("Real-novel character map", "armA_within"),
    ("Model-written story map", "armB_within"),
    ("Separator-anchor control", "armC_sep"),
    ("Preceding-sentence control", "armC_prevmean"),
    ("Chat reference", "chat_ref"),
]

PRETTY_DIRECTION = {
    "chat_ref->armA": "chat -> novel",
    "armA_within_lastpos->chat": "novel -> chat",
    "chat_ref->armB": "chat -> story-gen",
    "armB_within_lastpos->chat": "story-gen -> chat",
    "armA_within->armB": "novel -> story-gen",
    "armB_within->armA": "story-gen -> novel",
    "armC_sep->armA": "separator -> novel",
    "armA_within_lastpos->armC": "novel -> separator",
    "armC_prevmean->armA": "prev-sentence -> novel",
    "chat_ref->armA (spanmean)": "chat -> novel (span-mean X)",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=Path("eval_results/issue_931"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_931"))
    return ap.parse_args()


def _save(fig, fig_dir: Path, name: str, meta: dict) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    (fig_dir / f"{name}.meta.json").write_text(
        json.dumps({"metadata": common.metadata(SCRIPT, 0, 0), **meta}, indent=2, default=float)
    )
    plt.close(fig)
    print(f"[i931-figs] wrote {fig_dir / name}.png")


def _load(results_dir: Path, name: str) -> dict | None:
    p = results_dir / name
    return json.loads(p.read_text()) if p.exists() else None


def hero1_layer_curves(results_dir: Path, fig_dir: Path) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = paper_palette(len(REGIME_CELLS))
    plotted = []
    for (label, cell), color in zip(REGIME_CELLS, colors, strict=True):
        cells = _load(results_dir, f"cells_{cell}.json")
        nulls = _load(results_dir, f"nulls_{cell}.json")
        if cells is None:
            continue
        r2 = np.asarray(cells["r2_per_layer_obs"], dtype=float)
        xs = np.arange(len(r2))
        ax.plot(xs, r2, label=label, color=color, lw=1.8)
        if nulls is not None:
            nm = np.asarray(nulls["null_matrix"], dtype=float)
            if nm.size:
                ax.fill_between(
                    xs,
                    np.nanquantile(nm, 0.025, axis=0),
                    np.nanquantile(nm, 0.975, axis=0),
                    color=color,
                    alpha=0.15,
                    lw=0,
                )
        plotted.append(cell)
    for li in common.FROZEN_LAYERS:
        ax.axvline(li, color="gray", lw=0.6, ls=":", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Held-out pooled $R^2$ (group 5-fold)")
    ax.legend(fontsize=8, loc="best")
    _save(fig, fig_dir, "hero1_layer_curves", {"cells": plotted})


def hero2_transfer_heatmap(results_dir: Path, fig_dir: Path) -> None:
    tm = _load(results_dir, "transfer_matrix.json")
    if tm is None:
        return
    hl = tm["headline_layer"]
    rows = [
        r
        for r in tm["rows"]
        if r["layer"] == hl and r["application"] == "recentered" and r["power_matched"]
    ]
    if not rows:
        return
    # De-dup by direction+recipe (primary first).
    seen: dict[str, dict] = {}
    for r in rows:
        key = f"{r['direction']} ({r['x_recipe']})"
        seen.setdefault(key, r)
    labels = list(seen)
    fracs = np.asarray([seen[k]["fraction_of_ceiling"] for k in labels], dtype=float)
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(labels) + 1.6), layout="constrained")
    im = ax.imshow(fracs[:, None], cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        [
            f"{lbl}  [src n={seen[lbl]['n_train']}, denom n={seen[lbl]['denominator_n_train']}]"
            for lbl in labels
        ],
        fontsize=7.5,
    )
    ax.set_xticks([0])
    ax.set_xticklabels([f"recentered fraction of ceiling @ L{hl}"], fontsize=8)
    for i, k in enumerate(labels):
        v = seen[k]["fraction_of_ceiling"]
        ax.text(
            0,
            i,
            "n/a" if not np.isfinite(v) else f"{v:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if np.isfinite(v) and v < 0.6 else "black",
        )
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, fig_dir, "hero2_transfer_matrix", {"headline_layer": hl, "n_rows": len(labels)})


def per_group_scatter(results_dir: Path, fig_dir: Path) -> None:
    cells = _load(results_dir, "cells_armA_within.json")
    if cells is None or not cells.get("per_group_r2_headline"):
        return
    pg = cells["per_group_r2_headline"]
    names = sorted(pg, key=lambda k: pg[k])
    vals = [pg[k] for k in names]
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 0.28 * len(names) + 1.4))
    ax.scatter(vals, range(len(names)), s=18)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:32] for n in names], fontsize=7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel(f"Per-novel held-out $R^2$ @ L{cells['headline_layer']} (armA_within)")
    _save(fig, fig_dir, "per_novel_r2_scatter", {"n_groups": len(names)})


def spanmean_vs_lastpos(results_dir: Path, fig_dir: Path) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pairs = [("armA", "Real novels"), ("armB", "Model stories")]
    width = 0.35
    for j, (arm, _label) in enumerate(pairs):
        cm = _load(results_dir, f"cells_{arm}_within.json")
        cl = _load(results_dir, f"cells_{arm}_within_lastpos.json")
        if cm is None or cl is None:
            continue
        hl = cm["headline_layer"]
        ax.bar(j - width / 2, cm["r2_per_layer_obs"][hl], width, color="C0")
        ax.bar(j + width / 2, cl["r2_per_layer_obs"][hl], width, color="C1")
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p[1] for p in pairs])
    ax.set_ylabel("Held-out $R^2$ @ headline layer")
    ax.legend(["span-mean X", "boundary-token X (parent recipe)"], fontsize=8)
    _save(fig, fig_dir, "spanmean_vs_lastpos", {})


def power_curve_overlay(results_dir: Path, fig_dir: Path) -> None:
    pc = _load(results_dir, "power_curve_chat.json")
    if pc is None:
        return
    pts = [c for c in pc["curve"] if c.get("r2_per_layer")]
    if not pts:
        return
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ns = [c["n"] for c in pts]
    hl = min(common.HEADLINE_LAYER, len(pts[0]["r2_per_layer"]) - 1)
    ax.plot(ns, [c["r2_per_layer"][hl] for c in pts], "o-", label=f"chat_ref @ L{hl}")
    for cell, label in (("armA_within", "novel (full n)"), ("armB_within", "story-gen (full n)")):
        c = _load(results_dir, f"cells_{cell}.json")
        if c:
            ax.scatter([c["n"]], [c["r2_per_layer_obs"][c["headline_layer"]]], label=label)
    ax.set_xlabel("Training rows n")
    ax.set_ylabel("Held-out $R^2$")
    ax.legend(fontsize=8)
    _save(fig, fig_dir, "power_curve_overlay", {"ns": ns})


def matched_vs_fulln(results_dir: Path, fig_dir: Path) -> None:
    tm = _load(results_dir, "transfer_matrix.json")
    if tm is None:
        return
    hl = tm["headline_layer"]
    by_dir: dict[str, dict] = {}
    for r in tm["rows"]:
        if r["layer"] != hl or r["application"] != "recentered":
            continue
        d = by_dir.setdefault(f"{r['direction']} ({r['x_recipe']})", {})
        d["matched" if r["power_matched"] else "full"] = r["fraction_of_ceiling"]
    keys = [k for k, v in by_dir.items() if "matched" in v]
    if not keys:
        return
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 0.4 * len(keys) + 1.4))
    y = np.arange(len(keys))
    ax.scatter([by_dir[k]["matched"] for k in keys], y, label="matched power (PRIMARY)", s=22)
    fx = [(by_dir[k].get("full"), i) for i, k in enumerate(keys) if "full" in by_dir[k]]
    if fx:
        ax.scatter([v for v, _ in fx], [i for _, i in fx], label="full n (secondary)", s=22)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=7.5)
    ax.set_xlabel(f"Recentered fraction of ceiling @ L{hl}")
    ax.legend(fontsize=8)
    _save(fig, fig_dir, "matched_vs_fulln_fractions", {"n_directions": len(keys)})


def strict_vs_recentered(results_dir: Path, fig_dir: Path) -> None:
    tm = _load(results_dir, "transfer_matrix.json")
    if tm is None:
        return
    hl = tm["headline_layer"]
    by_dir: dict[str, dict] = {}
    for r in tm["rows"]:
        if r["layer"] != hl or not r["power_matched"]:
            continue
        by_dir.setdefault(f"{r['direction']} ({r['x_recipe']})", {})[r["application"]] = r[
            "transfer_r2"
        ]
    keys = sorted(by_dir)
    if not keys:
        return
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    rec = [by_dir[k].get("recentered", np.nan) for k in keys]
    st = [by_dir[k].get("strict", np.nan) for k in keys]
    ax.scatter(rec, st, s=20)
    for k, x, yv in zip(keys, rec, st, strict=True):
        ax.annotate(k, (x, yv), fontsize=6, alpha=0.8)
    lo = min(np.nanmin(rec), np.nanmin(st), 0)
    hi = max(np.nanmax(rec), np.nanmax(st), 0.05)
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.7, ls="--")
    ax.set_xlabel("Recentered transfer $R^2$ (primary)")
    ax.set_ylabel("Strict-frozen transfer $R^2$ (secondary)")
    _save(fig, fig_dir, "strict_vs_recentered", {"n_directions": len(keys)})


def delta_char_panel(results_dir: Path, fig_dir: Path) -> None:
    rows = []
    for arm in ("armA", "armB"):
        d = _load(results_dir, f"delta_char_{arm}.json")
        if d:
            rows.append((arm, d))
    if not rows:
        return
    set_paper_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (_arm, d) in enumerate(rows):
        ax.errorbar(
            [i],
            [d["delta_r2_char"]],
            yerr=[
                [max(0.0, d["delta_r2_char"] - d["delta_ci_lo"])],
                [max(0.0, d["delta_ci_hi"] - d["delta_r2_char"])],
            ],
            fmt="o",
            capsize=4,
        )
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["Real novels" if a == "armA" else "Model stories" for a, _ in rows])
    ax.set_ylabel(r"$\Delta R^2_{char}$ (correct $-$ swap), novel-level 95% CI")
    _save(fig, fig_dir, "delta_char", {"arms": [a for a, _ in rows]})


def main() -> int:
    args = parse_args()
    print("[phase=p4_figures] figures")
    hero1_layer_curves(args.results_dir, args.fig_dir)
    hero2_transfer_heatmap(args.results_dir, args.fig_dir)
    per_group_scatter(args.results_dir, args.fig_dir)
    spanmean_vs_lastpos(args.results_dir, args.fig_dir)
    power_curve_overlay(args.results_dir, args.fig_dir)
    matched_vs_fulln(args.results_dir, args.fig_dir)
    strict_vs_recentered(args.results_dir, args.fig_dir)
    delta_char_panel(args.results_dir, args.fig_dir)
    print("[i931-figs] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
