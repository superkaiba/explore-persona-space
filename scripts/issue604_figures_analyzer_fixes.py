"""Analyzer-pass figure fixes for task #604 (three replacements).

1. ``seed_stability`` — REPLACED: the Phase-C version plotted only the
   cross-seed key |cos| bars and dropped the write side, which is the
   finding (keys ~0.015, writes ~0.93). Two-strip paired plot.
2. ``write_match_panel`` — REPLACED: the Phase-C version was a 78-bar
   panel with unreadable slug tick labels. Two-panel: dial scatter vs
   dose with per-cell null spans + control bars vs the 0.5 bar.
3. ``i474_epoch_ladder`` — subtitle slope formatted ``{:+.3f}`` printed
   "+0.000"; re-render with 4 decimals and per-epoch units.

Reads eval_results/issue_604/{selectivity,write_match,rotation}.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "eval_results" / "issue_604"
FIG = PROJECT_ROOT / "figures" / "issue_604"


def fig_seed_stability(sel: dict) -> None:
    keys, writes = [], []
    for g in sel["seed_stability"]:
        for p in g["pairs"]:
            if p.get("key_abs_cos_band_mean") is not None:
                keys.append(p["key_abs_cos_band_mean"])
            if p.get("write_abs_cos_band_mean") is not None:
                writes.append(p["write_abs_cos_band_mean"])
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for x0, vals, role, label in (
        (0.0, keys, "primary", "key (top input direction)"),
        (1.0, writes, "accent", "write (top output direction)"),
    ):
        xs = x0 + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(xs, vals, s=14, alpha=0.65, color=paper_palette_role(role), label=label)
        med = float(np.median(vals))
        ax.plot([x0 - 0.22, x0 + 0.22], [med, med], color="black", lw=1.4)
        ax.annotate(f"median {med:.3f}", (x0 + 0.26, med), fontsize=8, va="center")
    ax.set_xticks([0.0, 1.0], ["key\n(top input direction)", "write\n(top output direction)"])
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cross-seed |cos|, layer-band mean")
    set_title_subtitle(
        ax,
        "Same data, different seed: the write direction is reproducible, the key is not",
        "one dot per seed pair within a training group (n = 69 pairs each)",
    )
    fig.tight_layout()
    savefig_paper(fig, "seed_stability", dir=FIG)
    plt.close(fig)


def fig_write_match(wm: dict) -> None:
    dial, controls = [], []
    for cell in wm["cells"]:
        if "per_source" in cell:
            for rec in cell["per_source"]:
                dial.append(
                    (
                        cell["dose"].get(rec["source"]),
                        rec["cos_abs"],
                        rec["null_p5"],
                        rec["null_p95"],
                    )
                )
        elif "variants" in cell:
            v = cell["variants"].get("same")
            if not isinstance(v, dict):
                continue
            if cell["line"] == "i521":
                comp = v.get("cos_pool_vs_U1_shared_direction")
                grp = "emergent-misalignment control"
            else:
                comp = v.get("source_cos")
                grp = "saturated marker endpoint"
            if comp is not None:
                controls.append((grp, cell["cell_id"], abs(comp)))
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    for dose, cos_abs, p5, p95 in dial:
        ax.plot([dose, dose], [p5, p95], color=paper_palette_role("neutral"), lw=1.0, alpha=0.5)
    ax.scatter(
        [d for d, *_ in dial],
        [c for _, c, *_ in dial],
        s=16,
        color=paper_palette_role("primary"),
        zorder=3,
        label="weight-space write vs source's measured shift",
    )
    ax.set_xlabel("realized implant depth (nat, re-measured per cell)")
    ax.set_ylabel("|cos(pooled write, measured shift)|")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left", fontsize=8)
    set_title_subtitle(
        ax,
        "The weight-space write does not match the measured shift",
        "grey spans = wrong-context null p5-p95 within each cell (n = 72 reads)",
    )
    order = [(g, c, v) for g, c, v in controls if g == "emergent-misalignment control"] + [
        (g, c, v) for g, c, v in controls if g != "emergent-misalignment control"
    ]
    xs = np.arange(len(order))
    colors = [
        paper_palette_role("control" if g == "emergent-misalignment control" else "baseline")
        for g, _, _ in order
    ]
    ax2.bar(xs, [v for _, _, v in order], color=colors)
    ax2.axhline(0.5, color="black", lw=1.0, ls="--")
    ax2.annotate("positive-control bar (0.5)", (0.02, 0.51), fontsize=7)
    ax2.set_xticks(
        xs,
        [
            ("EM " + c.rsplit("seed", 1)[-1])
            if g.startswith("emergent")
            else ("sat " + c.rsplit("seed", 1)[-1])
            for g, c, _ in order
        ],
        fontsize=7,
    )
    ax2.set_ylim(0, 0.6)
    ax2.set_ylabel("|cos| vs matched-seed shared direction")
    set_title_subtitle(ax2, "Positive control fails", "EM seeds + saturated endpoint, layer 14")
    fig.tight_layout()
    savefig_paper(fig, "write_match_panel", dir=FIG)
    plt.close(fig)


def fig_i474_ladder(rot: dict) -> None:
    lad = rot["i474_epoch_ladder"]
    reads = lad["reads"]
    agg = lad["aggregate"]
    by = {}
    for r in reads:
        by.setdefault((r["arm"], r["source"]), []).append((r["epoch"], r["delta_cos"]))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    seen = set()
    for (arm, _src), pts in sorted(by.items()):
        pts.sort()
        color = paper_palette_role("primary" if arm == "loc" else "accent")
        label = (
            ("contrastive arm" if arm == "loc" else "positives-only arm")
            if arm not in seen
            else None
        )
        seen.add(arm)
        ax.plot(
            [e for e, _ in pts], [d for _, d in pts], color=color, alpha=0.55, lw=1.0, label=label
        )
    ax.set_xticks([1, 2, 3, 5])
    ax.set_xlabel("training epochs")
    ax.set_ylabel("key rotation toward source-minus-others (Δ|cos|)")
    ax.legend(fontsize=8)
    set_title_subtitle(
        ax,
        "Does contrastive training rotate the key with epochs?",
        f"one line per source; paired slope difference {agg['mean']:+.4f}/epoch, "
        f"CI [{agg['ci_lo_mean']:+.4f}, {agg['ci_hi_mean']:+.4f}]",
    )
    fig.tight_layout()
    savefig_paper(fig, "i474_epoch_ladder", dir=FIG)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    sel = json.loads((OUT / "selectivity.json").read_text())
    wm = json.loads((OUT / "write_match.json").read_text())
    rot = json.loads((OUT / "rotation.json").read_text())
    fig_seed_stability(sel)
    fig_write_match(wm)
    fig_i474_ladder(rot)
    print("done:", FIG)


if __name__ == "__main__":
    main()
