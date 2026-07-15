"""Figures for issue #1090 fu4 (extended-dose-lr round, plan v6 §6).

Reads the fu4 aggregate (eval_results/issue_1090/fu4-extended-dose-lr/
fu4_ladders.json) and writes:

- HERO ``fu4_dose_lr_grid``: 3 panels (one per cell), x = optimizer step
  (5..75), y = primary rate (structural for formatting, judged for impolite),
  one line per lr rung, the registered 0.60-0.85 band shaded, the reused fu3
  base rate dashed; companion tf-margin markers at the selected rungs.
- EXPLORATORY dump under figures/issue_1090/fu4/: per-run margin-by-context
  curves, degeneracy-flag table, Tier-1 (max rung) vs Tier-2 agreement scatter.

Every figure uses plain-English condition names (no bare run ids in prose).
Runs VM-side after ``issue1090_fu4.py --phase judge-aggregate``; ``--ladders``
overrides the input path (smoke figures read the smoke mirror, never the
committed deliverables path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# Shared-VM thread caps (#847): apply env caps BEFORE matplotlib/numpy import.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LADDERS = ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr" / "fu4_ladders.json"

CELL_LABEL = {
    "fmt-pers": "list formatting, persona-trained (control)",
    "imp-pers": "impolite, persona-trained",
    "imp-conv": "impolite, WildChat-trained",
}
CELL_ORDER = ("fmt-pers", "imp-pers", "imp-conv")
LR_LABEL = {1e-05: "lr 1e-5 (usual)", 3e-05: "lr 3e-5", 0.0001: "lr 1e-4"}


def _lr_colors() -> dict[float, str]:
    pal = paper_palette_blog(3)
    return dict(zip(sorted(LR_LABEL), pal, strict=True))


def fig_dose_lr_grid(agg: dict, out_prefix: str, out_dir: str) -> None:
    """HERO: per-cell dose ladders, one line per lr, band + base overlays."""
    colors = _lr_colors()
    band = agg.get("band", [0.60, 0.85])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    for ax, cell_key in zip(axes, CELL_ORDER, strict=True):
        runs = [r for r in agg["runs"].values() if r["cell_key"] == cell_key]
        if not runs:
            ax.set_title(f"{CELL_LABEL[cell_key]}\nN/A — not run")
            continue
        ax.axhspan(band[0], band[1], color="0.9", zorder=0)
        base = (runs[0].get("base_tier2") or {}).get("rate")
        if base is not None:
            ax.axhline(base, ls="--", color="0.4", lw=1.2, label="base rate (reused fu3 read)")
        for r in sorted(runs, key=lambda x: x["lr"]):
            rates = r.get("rates_by_step") or {}
            if not rates:
                continue
            steps = sorted(int(s) for s in rates)
            ys = [rates[str(s)] for s in steps]
            ax.plot(
                steps,
                ys,
                marker="o",
                ms=3.5,
                lw=1.6,
                color=colors[r["lr"]],
                label=LR_LABEL[r["lr"]] + (" — DIVERGED" if r["status"] == "diverged" else ""),
            )
            sel = r.get("selection")
            t2 = r.get("tier2_trained")
            if sel is not None and t2 is not None:
                ax.scatter(
                    [sel["step"]],
                    [t2["rate"]],
                    marker="D",
                    s=42,
                    color=colors[r["lr"]],
                    edgecolor="black",
                    lw=0.6,
                    zorder=5,
                )
        ax.set_title(CELL_LABEL[cell_key])
        ax.set_xlabel("optimizer step (5 steps = 1 epoch)")
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("primary rate (structural / judged)")
    handles, labels = axes[1].get_legend_handles_labels()
    if not labels:
        handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    savefig_paper(fig, f"{out_prefix}/fu4_dose_lr_grid", dir=out_dir)
    plt.close(fig)


def fig_margin_at_selected(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-run tf-margin (trained vs base means) at the selected
    rung — the install-without-expression companion read."""
    rows = []
    for r in agg["runs"].values():
        m = r.get("margin") or {}
        if m.get("status") != "computed":
            continue
        rows.append((r["cell_key"], r["lr"], m.get("margin_base"), m.get("margin_trained")))
    if not rows:
        return
    colors = _lr_colors()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    xticks, xlabels = [], []
    x = 0
    for cell_key in CELL_ORDER:
        cell_rows = [r for r in rows if r[0] == cell_key]
        for _ck, lr, mb, mt in sorted(cell_rows, key=lambda t: t[1]):
            if mb is not None:
                ax.scatter([x], [mb], marker="s", color="0.5", s=30)
            if mt is not None:
                ax.scatter([x], [mt], marker="o", color=colors[lr], s=40)
            xticks.append(x)
            cell_short = CELL_LABEL[_ck].replace(" (control)", "").replace(", ", ",\n")
            xlabels.append(f"{cell_short}\n{LR_LABEL[lr].split()[1]}")
            x += 1
        x += 1
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("tf-margin (mean over eval contexts)")
    ax.set_title("fixed-pool margin at the selected rung (circles trained, squares base)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu4_margin_at_selected", dir=out_dir)
    plt.close(fig)


def fig_degeneracy_flags(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Exploratory: per-run count of degeneracy-flagged rungs (flag-only guard)."""
    labels, counts = [], []
    for r in sorted(agg["runs"].values(), key=lambda x: (x["cell_key"], x["lr"])):
        flags = [d for d in (r.get("degeneracy_by_step") or {}).values() if d.get("degenerate")]
        labels.append(f"{r['cell_key']} {LR_LABEL[r['lr']].split()[1]}")
        counts.append(len(flags))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.bar(range(len(labels)), counts, color=paper_palette_blog(1)[0])
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("rungs flagged degenerate")
    ax.set_title("degenerate-output guard flags per run (length/4-gram; flag-only)")
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu4_degeneracy_flags", dir=out_dir)
    plt.close(fig)


def fig_tier2_verdict(agg: dict, out_prefix: str, out_dir: str) -> None:
    """Per-arm Tier-2 confirmatory rate (Wilson 95% CI) with the registered
    0.60 band floor + 0.30 cut, reused base rates, and — for judged impolite
    arms — the worst-case judge-drop bound (every dropped completion scored
    non-impolite: k/200)."""
    colors = _lr_colors()
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    xticks, xlabels = [], []
    x = 0
    for cell_key in CELL_ORDER:
        cell_runs = sorted(
            (r for r in agg["runs"].values() if r["cell_key"] == cell_key),
            key=lambda r: r["lr"],
        )
        for r in cell_runs:
            t2 = r.get("tier2_trained") or {}
            rate, lo_hi = t2.get("rate"), t2.get("wilson95") or (None, None)
            if rate is None:
                continue
            ax.errorbar(
                [x],
                [rate],
                yerr=[[rate - lo_hi[0]], [lo_hi[1] - rate]],
                fmt="o",
                color=colors[r["lr"]],
                capsize=3,
                ms=6,
            )
            ax.text(x + 0.12, rate, f"{rate:.2f}", fontsize=7, va="center")
            if t2.get("mode") == "judged" and t2.get("n", 200) < 200:
                wc = t2["k"] / 200.0
                ax.scatter(
                    [x],
                    [wc],
                    marker="v",
                    facecolors="none",
                    edgecolors=colors[r["lr"]],
                    s=45,
                    linewidths=1.2,
                )
            base = (r.get("base_tier2") or {}).get("rate")
            if base is not None:
                ax.scatter([x], [base], marker="s", color="0.5", s=28, zorder=1)
            cell_short = CELL_LABEL[cell_key].replace(" (control)", "").replace(", ", ",\n")
            xlabels.append(f"{cell_short}\n{LR_LABEL[r['lr']].split()[1]}")
            xticks.append(x)
            x += 1
        x += 1
    ax.axhline(0.60, color="0.4", lw=0.9, ls="--")
    ax.axhline(0.30, color="0.7", lw=0.9, ls=":")
    ax.set_xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Tier-2 confirmatory rate")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title(
        "Tier-2 verdict reads (circles trained + Wilson 95% CI; squares base;\n"
        "open triangles worst-case judge-drop bound; dashes 0.60 band floor, dots 0.30 cut)"
    )
    fig.tight_layout()
    savefig_paper(fig, f"{out_prefix}/fu4_tier2_verdict", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="#1090 fu4 figures")
    ap.add_argument("--ladders", default=str(DEFAULT_LADDERS))
    ap.add_argument("--out-prefix", default="issue_1090/fu4")
    ap.add_argument("--out-dir", default="figures/")
    args = ap.parse_args()
    set_paper_style()
    agg = json.loads(Path(args.ladders).read_text())
    fig_dose_lr_grid(agg, args.out_prefix, args.out_dir)
    fig_margin_at_selected(agg, args.out_prefix, args.out_dir)
    fig_degeneracy_flags(agg, args.out_prefix, args.out_dir)
    fig_tier2_verdict(agg, args.out_prefix, args.out_dir)


if __name__ == "__main__":
    main()
