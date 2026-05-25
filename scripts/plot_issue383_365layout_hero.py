"""Hero plot for task #365: factor effects on source rate, leakage rate,
and **source-vs-leakage selectivity**.

Reads the 72 per-cell `metrics.json` files (committed on the
`task-365-implementation` branch) and computes, for each factor:

  * Δ source_rate (matched pair: factor=1 minus factor=0, holding source and
    the other four factors fixed)
  * Δ leakage_rate_full (same matched pair, mean over 23 non-source personas)
  * Δ selectivity = Δ source_rate − Δ leakage_rate_full (positive → factor
    lifts source faster than it lifts leakage; zero → factor is non-selective;
    negative → factor lifts leakage faster than source).

The hero has two panels:
  - Left: paired-bar chart of Δ source and Δ leakage per factor (absolute pp).
  - Right: bar chart of the selectivity Δ per factor (source-vs-leakage
    differential), with bootstrap CIs from the per-pair selectivity series.

Run from repo root:

    uv run python scripts/plot_issue365_hero.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL_ROOT = Path(".claude/worktrees/issue-383/eval_results/issue_383")

FACTOR_LABELS = {
    "A": "Long system prompt\n(vs short)",
    "B": "Long answer\n(vs short)",
    "C": "Neutral framing\n(vs persona)",
    "D": "Claude-written data\n(vs base-model)",
    "E": "Whole-completion loss\n(vs marker-focused)",
}

# Cell key is a 5-bit string A B C D E.
FACTOR_BIT = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

RNG = np.random.default_rng(42)


def load_cells() -> dict[tuple[str, str], dict]:
    """Return {(source, cell_key): {'source_rate', 'leakage_rate'}} for all
    cells where the metrics file exists and is non-failed."""
    out: dict[tuple[str, str], dict] = {}
    for cell_dir in sorted(EVAL_ROOT.glob("cell_*")):
        cell_key = cell_dir.name.removeprefix("cell_")
        if len(cell_key) != 5 or not all(c in "01" for c in cell_key):
            continue
        for source_dir in sorted(cell_dir.glob("source_*")):
            source = source_dir.name.removeprefix("source_")
            mfile = source_dir / "seed_42" / "metrics.json"
            if not mfile.exists():
                continue
            m = json.loads(mfile.read_text())
            if m.get("failed"):
                continue
            out[(source, cell_key)] = {
                "source_rate": float(m["source_substring_rate"]),
                "leakage_rate": float(m["leakage_rate_full"]),
            }
    return out


def matched_pairs(cells: dict, factor: str) -> list[dict]:
    """Return [{'source', 'd_src', 'd_lk', 'd_sel'} ...] for one matched pair
    per (source, fixed-other-bits) combination. Skips pairs where either arm
    is missing (e.g. A=0 × C=1 was excluded by design)."""
    bit = FACTOR_BIT[factor]
    by_source_others: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for (source, key), m in cells.items():
        others = key[:bit] + key[bit + 1 :]
        arm = key[bit]
        by_source_others[(source, others)][arm] = m

    pairs = []
    for (source, _others), arms in by_source_others.items():
        if "0" not in arms or "1" not in arms:
            continue
        m0, m1 = arms["0"], arms["1"]
        d_src = m1["source_rate"] - m0["source_rate"]
        d_lk = m1["leakage_rate"] - m0["leakage_rate"]
        pairs.append({"source": source, "d_src": d_src, "d_lk": d_lk, "d_sel": d_src - d_lk})
    return pairs


def percentile_boot_ci(values: list[float], n_boot: int = 1000) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean of `values`. Returns (lo, hi).
    Resamples WITH replacement; ignores cluster structure."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    boots = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def cluster_boot_ci(by_source: dict[str, list[float]], n_boot: int = 1000) -> tuple[float, float]:
    """Source-cluster bootstrap 95% CI on the pooled mean. Resamples the
    *source* clusters with replacement; for each resampled source, takes
    ALL its pair values; computes the mean across the collected pool.
    With n=3 source clusters this is low-resolution but it is what the
    `factor_effects.json` aggregator uses as one of its three CIs."""
    sources = list(by_source.keys())
    if not sources or not any(by_source.values()):
        return 0.0, 0.0
    n_clusters = len(sources)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = RNG.choice(n_clusters, size=n_clusters, replace=True)
        pool: list[float] = []
        for j in sampled:
            pool.extend(by_source[sources[j]])
        if not pool:
            boots[i] = np.nan
        else:
            boots[i] = float(np.mean(pool))
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return 0.0, 0.0
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def widest_ci(
    flat_values: list[float], by_source: dict[str, list[float]], n_boot: int = 1000
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) where (lo, hi) is the WIDEST of:
      * percentile bootstrap over per-pair values (ignores clusters)
      * source-cluster bootstrap (resamples sources, pools their pairs)
    Matches the `factor_effects.json` aggregator's "widest CI" convention."""
    arr = np.asarray(flat_values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    p_lo, p_hi = percentile_boot_ci(flat_values, n_boot=n_boot)
    c_lo, c_hi = cluster_boot_ci(by_source, n_boot=n_boot)
    lo = min(p_lo, c_lo)
    hi = max(p_hi, c_hi)
    return mean, lo, hi


def main() -> None:
    set_paper_style("blog")

    cells = load_cells()
    print(f"loaded {len(cells)} cell-source records")

    rows = []
    for code, label in FACTOR_LABELS.items():
        pairs = matched_pairs(cells, code)
        d_srcs = [p["d_src"] for p in pairs]
        d_lks = [p["d_lk"] for p in pairs]
        d_sels = [p["d_sel"] for p in pairs]

        src_by_source: dict[str, list[float]] = defaultdict(list)
        lk_by_source: dict[str, list[float]] = defaultdict(list)
        sel_by_source: dict[str, list[float]] = defaultdict(list)
        for p in pairs:
            src_by_source[p["source"]].append(p["d_src"])
            lk_by_source[p["source"]].append(p["d_lk"])
            sel_by_source[p["source"]].append(p["d_sel"])

        src_m, src_lo, src_hi = widest_ci(d_srcs, src_by_source)
        lk_m, lk_lo, lk_hi = widest_ci(d_lks, lk_by_source)
        sel_m, sel_lo, sel_hi = widest_ci(d_sels, sel_by_source)

        rows.append(
            {
                "code": code,
                "label": label,
                "n_pairs": len(pairs),
                "src_mean": 100 * src_m,
                "src_lo": 100 * src_lo,
                "src_hi": 100 * src_hi,
                "lk_mean": 100 * lk_m,
                "lk_lo": 100 * lk_lo,
                "lk_hi": 100 * lk_hi,
                "sel_mean": 100 * sel_m,
                "sel_lo": 100 * sel_lo,
                "sel_hi": 100 * sel_hi,
            }
        )

    for r in rows:
        print(
            f"  {r['code']}  n={r['n_pairs']:>3}  "
            f"Δsrc={r['src_mean']:+.2f} pp [{r['src_lo']:+.2f},{r['src_hi']:+.2f}]  "
            f"Δlk={r['lk_mean']:+.2f} pp [{r['lk_lo']:+.2f},{r['lk_hi']:+.2f}]  "
            f"Δsel={r['sel_mean']:+.2f} pp [{r['sel_lo']:+.2f},{r['sel_hi']:+.2f}]"
        )

    n = len(rows)
    y = np.arange(n)[::-1]
    bar_h = 0.36
    fig, (ax_l, ax_r) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.25},
    )

    src_color = paper_palette_role("primary")
    lk_color = paper_palette_role("accent")
    sel_color = paper_palette_role("baseline")

    # --- Left panel: paired Δsrc and Δlk ---
    for i, r in enumerate(rows):
        y_src = y[i] + bar_h / 2
        y_lk = y[i] - bar_h / 2
        ax_l.barh(
            y_src,
            r["src_mean"],
            height=bar_h,
            color=src_color,
            label="Δ source rate" if i == 0 else None,
        )
        ax_l.errorbar(
            r["src_mean"],
            y_src,
            xerr=[[r["src_mean"] - r["src_lo"]], [r["src_hi"] - r["src_mean"]]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
        ax_l.barh(
            y_lk,
            r["lk_mean"],
            height=bar_h,
            color=lk_color,
            label="Δ leakage rate" if i == 0 else None,
        )
        ax_l.errorbar(
            r["lk_mean"],
            y_lk,
            xerr=[[r["lk_mean"] - r["lk_lo"]], [r["lk_hi"] - r["lk_mean"]]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
    ax_l.axvline(0, color="#444444", lw=0.8)
    ax_l.set_yticks(y)
    ax_l.set_yticklabels([r["label"] for r in rows])
    ax_l.set_xlabel("Matched-pair Δ (percentage points)")
    ax_l.set_ylim(-0.7, n - 0.3)
    ax_l.grid(axis="x", lw=0.4, alpha=0.5)
    ax_l.legend(loc="lower right", frameon=False)
    ax_l.set_title("Absolute change: source vs leakage", fontsize=11)

    # --- Right panel: selectivity Δ ---
    for i, r in enumerate(rows):
        ax_r.barh(
            y[i],
            r["sel_mean"],
            height=0.55,
            color=sel_color,
        )
        ax_r.errorbar(
            r["sel_mean"],
            y[i],
            xerr=[[r["sel_mean"] - r["sel_lo"]], [r["sel_hi"] - r["sel_mean"]]],
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=3,
        )
    ax_r.axvline(0, color="#444444", lw=0.8)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])  # share left-panel labels
    ax_r.set_xlabel("Δ source rate − Δ leakage rate (pp)")
    ax_r.set_ylim(-0.7, n - 0.3)
    ax_r.grid(axis="x", lw=0.4, alpha=0.5)
    ax_r.set_title("Selectivity: how much faster source moves than leakage", fontsize=11)

    fig.suptitle(
        "Whole-completion loss is the one factor that lifts source rate far faster than bystander leakage",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.93,
        "Recipe-fix factor screen on Qwen2.5-7B-Instruct (72 LoRAs, 3 sources, seed 42, recipe-fix branch); CIs are wider of per-pair vs source-cluster bootstrap.",
        ha="center",
        fontsize=10,
        color="#444444",
    )
    fig.text(
        0.01,
        0.01,
        "source: eval_results/issue_383/cell_*/source_*/seed_42/metrics.json",
        fontsize=7,
        color="#888888",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    out_dir = Path("figures")
    savefig_paper(fig, "issue_383/hero_365_layout", dir=str(out_dir))
    plt.close(fig)
    print("saved figures/issue_383/hero_365_layout.png + .pdf + .meta.json")


if __name__ == "__main__":
    main()
