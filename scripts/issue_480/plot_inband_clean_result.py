"""Clean-result figures for issue #480 round-3 follow-up `inband-logprob-concordance`.

Produces two figures (PNG + PDF + .meta.json each) into an output dir:

1. ``hero_per_source_inband_logprob_vs_sycophancy`` — 6-facet per-source
   scatter of the in-band (step-20, sub-emission) per-cell marker log-prob
   delta against the frozen #411 sycophancy delta, mirroring the round-2
   hero's facet layout (blue = sycophancy-varying panels, grey =
   near-flat / descriptive-only).
2. ``regime_comparison_rho_families`` — per-source horizontal bars of the
   naive Spearman rho between each marker readout and the frozen sycophancy
   deltas, across both anchor regimes (firing-anchor: emission rate,
   log-prob delta, marker-logit delta, EOS-margin delta — recomputed from
   round 2's committed matrix; in-band: log-prob delta, EOS-margin delta).

Run from the issue-480-inband-logprob-concordance worktree (eval data lives
on that branch):

    uv run python scripts/issue_480/plot_inband_clean_result.py \
        --out-dir /path/to/repo-root/figures/issue_480/inband-logprob-concordance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as sps

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

INBAND_MATRIX = Path("eval_results/issue_480/inband-logprob-concordance/marker_delta_matrix.json")
FIRING_MATRIX = Path("eval_results/issue_480/band-stopped-anchor-rerun/marker_delta_matrix.json")

# Facet order matches the round-2 hero for cross-round comparability.
SOURCE_ORDER = [
    "software_engineer",
    "assistant",
    "comedian",
    "qwen_default",
    "kindergarten_teacher",
    "villain",
]
SOURCE_LABELS = {
    "software_engineer": "software engineer",
    "assistant": "assistant",
    "comedian": "comedian",
    "qwen_default": "Qwen default",
    "kindergarten_teacher": "kindergarten teacher",
    "villain": "villain",
}
# Recomputed from the frozen #411 join: cells with |sycophancy delta| > 0.10.
Y_ELIGIBLE = {"software_engineer", "assistant"}


def _load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    assert len(rows) == 138, f"expected 138 cross-cells in {path}, got {len(rows)}"
    return rows


def _rho(rows: list[dict], source: str, x_field: str) -> float:
    sub = [r for r in rows if r["source"] == source]
    assert len(sub) == 23, f"{source}: expected 23 bystander cells, got {len(sub)}"
    x = [r[x_field] for r in sub]
    y = [r["sycophancy_delta"] for r in sub]
    return float(sps.spearmanr(x, y).statistic)


def hero_figure(rows: list[dict], out_dir: Path) -> None:
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.6), sharey=False)

    blue = paper_palette_role("primary")
    grey = paper_palette_role("neutral")

    for ax, source in zip(axes.flat, SOURCE_ORDER):
        sub = [r for r in rows if r["source"] == source]
        x = [r["marker_delta"] for r in sub]
        xe = [r["marker_delta_se"] for r in sub]
        y = [r["sycophancy_delta"] for r in sub]
        ye = [r.get("sycophancy_delta_se", 0.0) or 0.0 for r in sub]
        eligible = source in Y_ELIGIBLE
        color = blue if eligible else grey
        ax.errorbar(
            x,
            y,
            xerr=xe,
            yerr=ye,
            fmt="o",
            ms=4.5,
            color=color,
            ecolor=color,
            elinewidth=0.9,
            capsize=0,
            alpha=0.85,
            linestyle="none",
        )
        tag = "sycophancy varies" if eligible else "sycophancy near-flat (descriptive)"
        ax.set_title(f"{SOURCE_LABELS[source]}\n({tag})", fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8)

    for ax in axes[1, :]:
        ax.set_xlabel("marker log P trained − base (nats)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("sycophancy leakage\n(trained − base)", fontsize=9)

    fig.subplots_adjust(top=0.78, hspace=0.62, wspace=0.30, left=0.09, right=0.97, bottom=0.11)
    fig.text(
        0.02,
        0.97,
        "In-band marker log-prob delta vs sycophancy leakage, per source, at sub-emission anchors",
        fontsize=12,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.02,
        0.935,
        "One point per bystander persona (n = 23 per panel); x error bars are per-cell SE over 50 "
        "probes; y error bars are per-bystander sycophancy SE.\nBlue = sources whose frozen "
        "sycophancy panel actually varies; grey = near-flat sycophancy (descriptive only). "
        "Zero of 7,200 generations contain the marker.",
        fontsize=8.5,
        color="#5A5A5A",
        ha="left",
        va="top",
    )
    savefig_paper(fig, "hero_per_source_inband_logprob_vs_sycophancy", dir=out_dir)
    plt.close(fig)


def regime_figure(inband_rows: list[dict], firing_rows: list[dict], out_dir: Path) -> None:
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False

    families = [
        ("emission rate", "firing", firing_rows, "emission_rate"),
        ("log-prob Δ", "firing", firing_rows, "marker_delta"),
        ("marker-logit Δ", "firing", firing_rows, "delta_z_marker"),
        ("EOS-margin Δ", "firing", firing_rows, "eos_margin_delta"),
        ("log-prob Δ", "inband", inband_rows, "marker_delta"),
        ("EOS-margin Δ", "inband", inband_rows, "eos_margin_delta"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    firing_color = paper_palette_role("baseline")
    inband_color = paper_palette_role("primary")

    for ax, source in zip(axes, ["software_engineer", "assistant"]):
        labels, values, colors = [], [], []
        for name, regime, rows, x_field in families:
            rho = _rho(rows, source, x_field)
            prefix = "firing anchor" if regime == "firing" else "in-band anchor"
            labels.append(f"{prefix} · {name}")
            values.append(rho)
            colors.append(firing_color if regime == "firing" else inband_color)
        ypos = list(range(len(labels)))[::-1]
        ax.barh(ypos, values, color=colors, height=0.62)
        ax.axvline(0.0, color="#999999", lw=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_title(SOURCE_LABELS[source], fontsize=10.5, loc="left")
        ax.set_xlim(-0.75, 0.75)
        ax.tick_params(labelsize=8.5)
        ax.set_xlabel("Spearman rho vs frozen sycophancy delta (n = 23)", fontsize=9)

    fig.subplots_adjust(top=0.80, wspace=0.55, left=0.20, right=0.97, bottom=0.14)
    fig.text(
        0.02,
        0.95,
        "Marker slot readouts invert at firing anchors and flip positive at sub-emission anchors",
        fontsize=12,
        fontweight="semibold",
        ha="left",
    )
    fig.text(
        0.02,
        0.895,
        "Naive per-source Spearman rho vs the frozen sycophancy deltas, on the two panels where "
        "sycophancy varies. Orange = firing-anchor (step 40) readouts; blue = in-band (step 20, "
        "sub-emission) readouts.",
        fontsize=8.5,
        color="#5A5A5A",
        ha="left",
        va="top",
    )
    savefig_paper(fig, "regime_comparison_rho_families", dir=out_dir)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    inband_rows = _load_rows(INBAND_MATRIX)
    firing_rows = _load_rows(FIRING_MATRIX)
    hero_figure(inband_rows, args.out_dir)
    regime_figure(inband_rows, firing_rows, args.out_dir)
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
