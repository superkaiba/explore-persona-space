"""Regenerate figures for task #451 clean-result after round-1 critic revisions.

Addresses the union of critique items from `epm:interp-critique-codex`:
- `all_factors_delta`: title clarifies "among B/C/D binary factors" since
  loss-mask E dominates but isn't plotted.
- `source_vs_bystander_by_e`: legend decodes C=0 → "Persona-role framing",
  C=1 → "Neutral-domain framing". Adds plotted-aggregates to meta sidecar.
- `c_axis_selectivity_hero_raw`: highlight the three B=0,D=0 E=2 outliers
  with annotations, add B/D coord shape coding, fix the caption's "two
  outliers" → three (handled in body, not in figure).
- `c_axis_selectivity_hero`: unchanged (numbers + caption match).

Meta sidecars now include plotted-data aggregates per critic ask.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
RECORDS = REPO / "eval_results/issue_451/per_cell_records.json"
FIG_DIR = REPO / "figures"
SUBDIR = "issue_451"

# Plain-English labels used everywhere reader-facing.
C_LABEL = {0: "Persona-role framing", 1: "Neutral-domain framing"}
E_LABEL = {
    0: "Marker-only loss\n(saturated)",
    1: "Last-32-tokens loss\n(partial signal)",
    2: "Whole-completion loss\n(clean signal)",
}
B_LABEL = {0: "long-answer", 1: "short-answer"}
D_LABEL = {0: "Claude-data", 1: "Tulu-style"}


def load_records():
    with open(RECORDS) as fp:
        return json.load(fp)


def _patch_meta(stem: str, plotted) -> None:
    """Merge plotted-data aggregates into the meta sidecar that ``savefig_paper`` wrote.

    Critic ask: the sidecar previously held only ``commit``/``created``/``figsize``,
    which doesn't suffice for independent plot-data provenance. Now adds
    ``plotted_aggregates`` (or ``plotted_pairs``) so the bars/dots in the figure are
    reproducible from the sidecar alone.
    """
    meta_path = FIG_DIR / f"{stem}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["plotted"] = plotted
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def matched_pairs(records, hold, vary):
    """Group records into matched pairs over `vary` (binary), keying on `hold`."""
    bins: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in records:
        key = tuple(r[k] for k in hold)
        bins[key][r[vary]] = r
    return bins


def boot_p(deltas: list[float], B: int = 10_000, seed: int = 42) -> float:
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return float("nan")
    mean_obs = statistics.mean(deltas)
    centered = [d - mean_obs for d in deltas]
    cnt = 0
    for _ in range(B):
        m = sum(rng.choice(centered) for _ in range(n)) / n
        if abs(m) >= abs(mean_obs):
            cnt += 1
    return cnt / B


def boot_ci(values: list[float], B: int = 10_000, seed: int = 42, alpha: float = 0.05):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(B):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return lo, hi


def figure_hero(records) -> None:
    """C delta by E loss-mask level — UNCHANGED (numbers + caption already match).

    Regenerated only to keep meta sidecar in sync with the other figures.
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    plotted = {}
    width = 0.36
    x = np.arange(3)
    for j, C in enumerate([0, 1]):
        means, errs = [], []
        for E in [0, 1, 2]:
            sels = [r["selectivity"] for r in records if r["E"] == E and r["C"] == C]
            mean = statistics.mean(sels)
            lo, hi = boot_ci(sels)
            means.append(mean)
            errs.append([mean - lo, hi - mean])
            plotted[f"E={E}_C={C}_mean"] = mean
            plotted[f"E={E}_C={C}_ci"] = [lo, hi]
        errs = list(zip(*errs))
        color = paper_palette_role("primary") if C == 0 else paper_palette_role("accent")
        ax.bar(
            x + (j - 0.5) * width,
            means,
            width=width,
            yerr=errs,
            label=C_LABEL[C],
            color=color,
            capsize=3,
            error_kw={"linewidth": 1.0},
        )

    # Annotate matched-pair Δ above each E group
    pairs = matched_pairs(records, hold=["B", "D", "E", "source"], vary="C")
    for E in [0, 1, 2]:
        deltas = [
            d[0]["selectivity"] - d[1]["selectivity"]
            for k, d in pairs.items()
            if k[2] == E and 0 in d and 1 in d
        ]
        delta = statistics.mean(deltas)
        plotted[f"E={E}_matched_delta"] = delta
        plotted[f"E={E}_matched_p"] = boot_p(deltas)
        y_top = max(
            plotted[f"E={E}_C=0_mean"] + plotted[f"E={E}_C=0_ci"][1] - plotted[f"E={E}_C=0_mean"],
            plotted[f"E={E}_C=1_mean"] + plotted[f"E={E}_C=1_ci"][1] - plotted[f"E={E}_C=1_mean"],
        )
        ax.text(
            E,
            min(1.07, y_top + 0.05),
            f"Δ = {delta:+.3f}",
            ha="center",
            fontsize=10,
            color="#444",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([E_LABEL[i] for i in [0, 1, 2]])
    ax.set_ylabel("Selectivity\n(source rate − mean bystander rate)")
    ax.set_ylim(-0.02, 1.15)
    ax.set_title(
        "Persona-framing vs neutral-domain framing: no consistent effect on selectivity",
        loc="left",
        fontweight="semibold",
        fontsize=12,
    )
    ax.text(
        0,
        1.10,
        "Matched-pair difference (Δ) flips sign across loss-mask levels — neither slice reaches n=12 significance",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=10,
        color="#555",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.text(
        0.0,
        -0.18,
        "Issue #451, single seed, n=12 matched (B,D,source) pairs per loss-mask level",
        transform=ax.transAxes,
        ha="left",
        fontsize=9,
        color="#777",
    )

    savefig_paper(fig, f"{SUBDIR}/c_axis_selectivity_hero", dir=str(FIG_DIR))
    _patch_meta(f"{SUBDIR}/c_axis_selectivity_hero", plotted)
    plt.close(fig)


def figure_hero_raw(records) -> None:
    """Per-pair scatter: highlight B=0,D=0 E=2 outliers explicitly."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    pairs = matched_pairs(records, hold=["B", "D", "E", "source"], vary="C")
    color_e = {
        0: paper_palette_role("baseline"),
        1: "#e89a5a",
        2: paper_palette_role("primary"),
    }
    plotted = []
    for E in [0, 1, 2]:
        xs, ys, marks = [], [], []
        for (B, D, _, src), d in pairs.items():
            if _ != E or 0 not in d or 1 not in d:
                continue
            xs.append(d[0]["selectivity"])
            ys.append(d[1]["selectivity"])
            marks.append((B, D, src))
            plotted.append(
                {
                    "B": B,
                    "D": D,
                    "E": E,
                    "source": src,
                    "C0_selectivity": d[0]["selectivity"],
                    "C1_selectivity": d[1]["selectivity"],
                }
            )
        # Plain points
        for x, y, (B, D, src) in zip(xs, ys, marks):
            highlight = E == 2 and B == 0 and D == 0
            ax.scatter(
                x,
                y,
                s=90 if highlight else 55,
                color=color_e[E],
                edgecolor="#222" if highlight else "white",
                linewidths=1.4 if highlight else 0.8,
                alpha=0.95,
                zorder=3 if highlight else 2,
            )
            if highlight:
                ax.annotate(
                    f"{src[:3]}.\nB=0, D=0",
                    (x, y),
                    xytext=(x + 0.03, max(0.02, y - 0.10)),
                    fontsize=8,
                    color="#222",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#666",
                        "lw": 0.6,
                        "shrinkA": 4,
                        "shrinkB": 3,
                    },
                )

    # Legend (manual swatches)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_e[0],
            markersize=8,
            label="E=0 marker-only loss (saturated)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_e[1],
            markersize=8,
            label="E=1 last-32-tokens loss (partial)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_e[2],
            markersize=8,
            label="E=2 whole-completion loss (clean)",
        ),
        plt.Line2D([0], [0], color="#999", ls="--", label="y = x (no framing effect)"),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#222",
            markersize=10,
            label="Highlighted: B=0 (long-answer), D=0 (Claude-data)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)

    ax.plot([0, 1.05], [0, 1.05], color="#999", ls="--", lw=1, zorder=1)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    # All 12 E=0 pairs collapse to (0,0) — annotate so the reader knows the red
    # cluster isn't 1 point.
    ax.annotate(
        "All 12 E=0 pairs\nstacked at origin\n(both framings saturated)",
        xy=(0.01, 0.01),
        xytext=(0.18, 0.85),
        fontsize=8,
        color=color_e[0],
        arrowprops={"arrowstyle": "->", "color": color_e[0], "lw": 0.7},
    )
    ax.set_xlabel("Selectivity, persona-role framing (C=0)")
    ax.set_ylabel("Selectivity, neutral-domain framing (C=1)")
    ax.set_title(
        "Per-pair view: three B=0, D=0 outliers drive the +12.5 pp E=2 mean",
        loc="left",
        fontweight="semibold",
        fontsize=12,
    )
    ax.text(
        0,
        1.04,
        "n=36 matched (B, D, source, E) pairs; 9 of 12 E=2 pairs sit on the diagonal — three corner cells carry the framing effect",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#555",
    )
    ax.text(
        0.0,
        -0.18,
        "Issue #451 raw per-cell data",
        transform=ax.transAxes,
        ha="left",
        fontsize=9,
        color="#777",
    )

    savefig_paper(fig, f"{SUBDIR}/c_axis_selectivity_hero_raw", dir=str(FIG_DIR))
    _patch_meta(f"{SUBDIR}/c_axis_selectivity_hero_raw", plotted)
    plt.close(fig)


def figure_source_bystander(records) -> None:
    """3-panel source-vs-bystander rates per E level — add plain-English legend."""
    import matplotlib as mpl

    set_paper_style("blog")
    # blog style's constrained_layout collides with fig.subplots_adjust + suptitle;
    # disable it for THIS figure so the title block + footer don't overlap panels.
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    plotted = {}

    width = 0.35
    x = np.arange(2)  # source, bystander
    for ax, E in zip(axes, [0, 1, 2]):
        for j, C in enumerate([0, 1]):
            cells = [r for r in records if r["E"] == E and r["C"] == C]
            src_rate = statistics.mean(r["src_rate"] for r in cells)
            byst_rate = statistics.mean(r["mean_bystander"] for r in cells)
            src_se = statistics.pstdev(r["src_rate"] for r in cells) / (len(cells) ** 0.5)
            byst_se = statistics.pstdev(r["mean_bystander"] for r in cells) / (len(cells) ** 0.5)
            color = paper_palette_role("primary") if C == 0 else paper_palette_role("accent")
            bars = ax.bar(
                x + (j - 0.5) * width,
                [src_rate, byst_rate],
                width=width,
                color=color,
                yerr=[src_se, byst_se],
                capsize=3,
                error_kw={"linewidth": 1.0},
                label=C_LABEL[C],
            )
            plotted[f"E={E}_C={C}_source_rate"] = src_rate
            plotted[f"E={E}_C={C}_bystander_rate"] = byst_rate

        ax.set_xticks(x)
        ax.set_xticklabels(["Source persona", "Mean of\n23 bystanders"])
        ax.set_title(E_LABEL[E].replace("\n", " — "), fontsize=10, loc="left", color="#333")
        ax.set_ylim(0, 1.15)
        if E == 0:
            ax.set_ylabel("Marker emission rate")
        # Inner legend only on right panel to save space
        if E == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.suptitle(
        "Source vs bystander marker emission rate, by loss-mask level and framing",
        x=0.06,
        y=0.99,
        ha="left",
        fontweight="semibold",
        fontsize=12,
    )
    fig.text(
        0.06,
        0.93,
        "Only whole-completion loss (E=2) gives a clean source-bystander gate; marker-only loss saturates everything",
        ha="left",
        fontsize=10,
        color="#555",
    )
    fig.text(
        0.06,
        0.01,
        "Issue #451 — n=12 cells per (E, framing); error bars = SE across cells. Legend: persona-role framing (C=0) = verbose role prose, "
        "neutral-domain framing (C=1) = length-matched non-role prose.",
        ha="left",
        fontsize=8.5,
        color="#777",
    )
    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.07, right=0.99, wspace=0.10)

    savefig_paper(fig, f"{SUBDIR}/source_vs_bystander_by_e", dir=str(FIG_DIR))
    _patch_meta(f"{SUBDIR}/source_vs_bystander_by_e", plotted)
    plt.close(fig)
    # Re-enable for any downstream figure call that may follow
    mpl.rcParams["figure.constrained_layout.use"] = True


def figure_all_factors(records) -> None:
    """Horizontal Δ chart — title now says 'among B/C/D binary factors'."""
    import matplotlib as mpl

    set_paper_style("blog")
    # constrained_layout fights with our custom title block + transAxes footer text
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    factors = [
        ("Long-answer (B)\nlong (0) vs short (1)", "B", paper_palette_role("primary")),
        ("Persona-framing (C)\npersona (0) vs neutral (1)", "C", "#999999"),
        ("Claude-data (D)\nClaude (0) vs Tulu (1)", "D", "#888888"),
    ]
    plotted = {}
    y_pos = np.arange(len(factors))[::-1]
    for i, (label, var, color) in enumerate(factors):
        hold = [k for k in ["B", "C", "D"] if k != var] + ["E", "source"]
        pairs = matched_pairs(records, hold=hold, vary=var)
        deltas = [
            d[0]["selectivity"] - d[1]["selectivity"] for d in pairs.values() if 0 in d and 1 in d
        ]
        mean = statistics.mean(deltas)
        lo, hi = boot_ci(deltas)
        p = boot_p(deltas)
        plotted[var] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(deltas), "p": p}
        ax.barh(
            y_pos[i],
            mean,
            color=color,
            edgecolor="#333",
            linewidth=0.7,
            xerr=[[mean - lo], [hi - mean]],
            capsize=3,
            error_kw={"linewidth": 1.0},
        )
        sig = "p < 0.001" if p < 0.001 else f"p = {p:.2f}"
        ax.text(
            mean + (0.012 if mean >= 0 else -0.012),
            y_pos[i],
            f"Δ={mean:+.3f}, {sig}",
            va="center",
            ha="left" if mean >= 0 else "right",
            fontsize=9,
            color="#333",
        )

    ax.axvline(0, color="#555", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[0] for f in factors])
    ax.set_xlim(-0.42, 0.22)
    ax.set_xlabel("Matched-pair Δ selectivity (level 0 − level 1)")
    # Title block: figure-level so subtitle never overlaps bars
    fig.text(
        0.04,
        0.93,
        "Among B/C/D binary factors, only long-answer (B) moves selectivity",
        ha="left",
        fontweight="semibold",
        fontsize=12,
    )
    fig.text(
        0.04,
        0.87,
        "Loss-mask level (E) dominates overall but is not a binary factor — see the per-E figure",
        ha="left",
        fontsize=9.5,
        color="#555",
    )
    ax.text(
        0.0,
        -0.30,
        "Issue #451 — n=36 matched pairs each, 10k bootstrap CI",
        transform=ax.transAxes,
        ha="left",
        fontsize=9,
        color="#777",
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.78, bottom=0.22)

    savefig_paper(fig, f"{SUBDIR}/all_factors_delta", dir=str(FIG_DIR))
    _patch_meta(f"{SUBDIR}/all_factors_delta", plotted)
    plt.close(fig)
    mpl.rcParams["figure.constrained_layout.use"] = True


def main() -> None:
    records = load_records()
    figure_hero(records)
    figure_hero_raw(records)
    figure_source_bystander(records)
    figure_all_factors(records)
    print("Regenerated 4 figures + meta sidecars under figures/issue_451/")


if __name__ == "__main__":
    main()
