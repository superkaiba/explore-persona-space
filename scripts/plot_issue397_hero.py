"""Hero plot for task #397: recipe-factor selectivity screen re-run with
single-token marker ※ at lr=1e-4 (seed 42).

Two panels:
  - Left: per-E-level source rate vs mean bystander leakage rate (24 cells per
    E level; persona-framed only because the persona-framing C=1 cells all
    failed at Pass 1). The E=0 (marker+EOT loss) cells saturate at 1.0 for
    both source and bystander — the marker fires on every persona regardless
    of training context. E=1 (tail-32 loss) is partial. Only E=2 (whole-
    completion loss) preserves selectivity.
  - Right: per-factor matched-pair selectivity Δ (A, B, D, E2-vs-E0) compared
    against the parent #383 single-seed Δs at the original [ZLT] marker +
    lr=1e-5 recipe. Signs replicate for all four available factors; ordering
    matches except A↔B.

Run from repo root:

    uv run python scripts/plot_issue397_hero.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

EVAL_ROOT = Path("eval_results/issue_397")


def load_cells():
    """Return list of {cell, src, a, b, c, d, e, src_rate, bys_rate, sel}."""
    out = []
    for cell_dir in sorted(EVAL_ROOT.glob("cell_*")):
        cell = cell_dir.name.removeprefix("cell_")
        if len(cell) != 5 or not all(c in "012" for c in cell):
            continue
        a, b, c, d, e = (int(x) for x in cell)
        for src_dir in sorted(cell_dir.glob("source_*")):
            src = src_dir.name.removeprefix("source_")
            m_path = src_dir / "seed_42" / "metrics.json"
            if not m_path.exists():
                continue
            m = json.loads(m_path.read_text())
            src_rate = m["personas"][src]["substring_rate"]
            bys_rates = [m["personas"][p]["substring_rate"] for p in m["personas"] if p != src]
            bys_mean = sum(bys_rates) / len(bys_rates)
            out.append(
                {
                    "cell": cell,
                    "src": src,
                    "a": a,
                    "b": b,
                    "c": c,
                    "d": d,
                    "e": e,
                    "src_rate": src_rate,
                    "bys_rate": bys_mean,
                    "sel": src_rate - bys_mean,
                }
            )
    return out


def matched_factor_delta(records, factor):
    """Per-factor matched-pair selectivity Δ within the C=0 stratum.

    Returns (list_of_pair_deltas, n_pairs).
    """
    idx = {(r["src"], r["a"], r["b"], r["c"], r["d"], r["e"]): r for r in records}
    pairs = []
    sources = ["librarian", "programmer", "surgeon"]
    if factor == "A":
        for src in sources:
            for B in (0, 1):
                for D in (0, 1):
                    for E in (0, 1, 2):
                        k0 = (src, 0, B, 0, D, E)
                        k1 = (src, 1, B, 0, D, E)
                        if k0 in idx and k1 in idx:
                            pairs.append(idx[k1]["sel"] - idx[k0]["sel"])
    elif factor == "B":
        for src in sources:
            for A in (0, 1):
                for D in (0, 1):
                    for E in (0, 1, 2):
                        k0 = (src, A, 0, 0, D, E)
                        k1 = (src, A, 1, 0, D, E)
                        if k0 in idx and k1 in idx:
                            pairs.append(idx[k1]["sel"] - idx[k0]["sel"])
    elif factor == "D":
        for src in sources:
            for A in (0, 1):
                for B in (0, 1):
                    for E in (0, 1, 2):
                        k0 = (src, A, B, 0, 0, E)
                        k1 = (src, A, B, 0, 1, E)
                        if k0 in idx and k1 in idx:
                            pairs.append(idx[k1]["sel"] - idx[k0]["sel"])
    elif factor == "E":  # E2 vs E0
        for src in sources:
            for A in (0, 1):
                for B in (0, 1):
                    for D in (0, 1):
                        k0 = (src, A, B, 0, D, 0)
                        k1 = (src, A, B, 0, D, 2)
                        if k0 in idx and k1 in idx:
                            pairs.append(idx[k1]["sel"] - idx[k0]["sel"])
    return pairs


def bootstrap_mean_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int((1 - ci) / 2 * n_boot)]
    hi = boots[int((1 + ci) / 2 * n_boot) - 1]
    return sum(values) / n, lo, hi


def main():
    set_paper_style("blog")
    records = load_cells()
    assert len(records) == 72, f"expected 72 records, got {len(records)}"

    # Left panel: per-E-level source rate vs bystander rate (mean over 24 cells per E)
    e_labels = [
        "Marker-only loss\n(loss on ※ + EOT)",
        "Tail-32 loss\n(loss on last ~32 tokens)",
        "Whole-completion loss\n(loss on all ~600 tokens)",
    ]
    e_src_means = []
    e_bys_means = []
    e_sel_means = []
    for E in (0, 1, 2):
        rs = [r for r in records if r["e"] == E]
        e_src_means.append(sum(r["src_rate"] for r in rs) / len(rs))
        e_bys_means.append(sum(r["bys_rate"] for r in rs) / len(rs))
        e_sel_means.append(sum(r["sel"] for r in rs) / len(rs))

    # Right panel: per-factor matched-pair selectivity Δ vs #383
    factors_383 = {"A": 33.6, "B": 27.8, "D": 11.2, "E": 41.7}
    factor_labels_full = {
        "A": "Long system\nprompt",
        "B": "Long answer\n",
        "D": "Claude-written\ntraining data",
        "E": "Whole-completion\nvs marker-only\nloss",
    }
    factor_data = {}
    for f in ("A", "B", "D", "E"):
        pairs = matched_factor_delta(records, f)
        m, lo, hi = bootstrap_mean_ci(pairs)
        factor_data[f] = {
            "397_mean": m * 100,
            "397_lo": lo * 100,
            "397_hi": hi * 100,
            "383": factors_383[f],
            "n": len(pairs),
        }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    # LEFT PANEL — per-E saturation
    ax = axes[0]
    x = np.arange(3)
    w = 0.35
    src_color = paper_palette_role("primary")
    bys_color = paper_palette_role("baseline")
    sel_color = paper_palette_role("accent")
    bars1 = ax.bar(
        x - w / 2, [r * 100 for r in e_src_means], w, label="Source persona", color=src_color
    )
    bars2 = ax.bar(
        x + w / 2,
        [r * 100 for r in e_bys_means],
        w,
        label="Mean of 23 bystander personas",
        color=bys_color,
    )
    # Overlay selectivity Δ as a line
    ax2 = ax.twinx()
    ax2.plot(
        x,
        [r * 100 for r in e_sel_means],
        color=sel_color,
        marker="D",
        linewidth=2.2,
        markersize=8,
        label="Selectivity Δ",
    )
    ax2.set_ylabel("Selectivity Δ\n(source − bystander, pp)", color=sel_color)
    ax2.tick_params(axis="y", labelcolor=sel_color)
    ax2.set_ylim(-5, 105)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(e_labels, fontsize=9)
    ax.set_ylabel("Marker emission rate (%)")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45), fontsize=9)
    ax2.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=9)
    set_title_subtitle(
        ax,
        "Marker-only loss saturates the persona panel",
        "Only whole-completion loss preserves selectivity (n = 24 cells / level, persona-framed recipes only)",
    )

    # RIGHT PANEL — per-factor Δ vs #383
    ax = axes[1]
    fs = ["A", "B", "D", "E"]
    x = np.arange(len(fs))
    w = 0.4
    new_means = [factor_data[f]["397_mean"] for f in fs]
    new_errs_lo = [factor_data[f]["397_mean"] - factor_data[f]["397_lo"] for f in fs]
    new_errs_hi = [factor_data[f]["397_hi"] - factor_data[f]["397_mean"] for f in fs]
    old_vals = [factor_data[f]["383"] for f in fs]
    new_color = paper_palette_role("primary")
    old_color = paper_palette_role("baseline")
    ax.bar(
        x - w / 2,
        old_vals,
        w,
        label="Parent ([ZLT] marker, lr=1e-5)",
        color=old_color,
        alpha=0.7,
    )
    ax.bar(
        x + w / 2,
        new_means,
        w,
        yerr=[new_errs_lo, new_errs_hi],
        label="This run (※ marker, lr=1e-4)",
        color=new_color,
        capsize=4,
    )
    ax.axhline(0, color="#999", linewidth=0.8)
    ax.set_xticks(x)
    # Append n= to each label so it doesn't collide with the x-axis ticks
    ax.set_xticklabels(
        [f"{factor_labels_full[f]}\n(n={factor_data[f]['n']})" for f in fs],
        fontsize=9,
    )
    ax.set_ylabel("Selectivity Δ (pp)\nsource − bystander rate gap")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(-5, 105)
    set_title_subtitle(
        ax,
        "Per-factor selectivity Δ vs parent run",
        "Signs replicate for all 4 available factors (persona-framing flip is unmeasured)",
    )

    plt.tight_layout()
    savefig_paper(fig, "issue_397/hero", dir="figures/")
    plt.close(fig)
    print("Saved figures/issue_397/hero.png + .pdf + .meta.json")
    print("Per-factor results:")
    for f in fs:
        d = factor_data[f]
        print(
            f"  {f}: #397 Δsel={d['397_mean']:+.1f} pp [{d['397_lo']:+.1f}, {d['397_hi']:+.1f}] (n={d['n']});"
            f" #383 Δsel={d['383']:+.1f} pp"
        )
    print(f"Per-E: src={e_src_means}, bys={e_bys_means}, sel={e_sel_means}")


if __name__ == "__main__":
    main()
