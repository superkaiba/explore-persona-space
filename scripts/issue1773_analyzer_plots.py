"""Analyzer supporting figures for issue #1773 (SAE feature description pipeline).

Reads the committed Phase-4/Phase-3 artifacts under ``eval_results/issue_1773/``
and renders two supporting figures:

1. ``battery_score_distributions`` — the low-level per-feature score
   distributions behind the scorecard aggregates (detection / fuzzing /
   discrimination; real vs shuffled-label vs random-init arms).
2. ``kappa_vs_prevalence`` — per-axis inter-draw Fleiss kappa against the
   modal-label prevalence, plus the majority-vote ``unresolved`` rate per axis
   (from the joined feature table).

Usage (from the issue-1773 worktree root)::

    uv run python scripts/issue1773_analyzer_plots.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

# CRITICAL: load_dotenv() BEFORE importing numpy/matplotlib — the shared-VM
# thread caps (#847) bind in-process only when set before the first BLAS/
# torch import freezes the pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "eval_results" / "issue_1773" / "validation"
LABELS = ROOT / "eval_results" / "issue_1773" / "labels"
TABLE = ROOT / "eval_results" / "issue_1773" / "feature_table_v1.jsonl"
FIGDIR = ROOT / "figures" / "issue_1773"

AXES = ["abstraction", "speaker_property", "content_type", "functional_role", "interpretable"]


def _battery_rows() -> list[dict]:
    rows: list[dict] = []
    for name in ("detection_fuzzing.jsonl", "discrimination.jsonl"):
        with (VAL / name).open() as fh:
            rows.extend(json.loads(line) for line in fh)
    return rows


def plot_battery_distributions() -> None:
    import matplotlib.pyplot as plt

    rows = _battery_rows()

    def scores(arm: str, battery: str) -> np.ndarray:
        vals = [
            r["score"]
            for r in rows
            if r["arm"] == arm
            and r["battery"] == battery
            and not (isinstance(r["score"], float) and math.isnan(r["score"]))
        ]
        return np.asarray(vals, dtype=float)

    panels = [
        (
            "detection",
            [("real", "real"), ("shuffled", "shuffled label"), ("randinit", "random init")],
        ),
        ("fuzzing", [("real", "real"), ("shuffled", "shuffled label")]),
        ("discrimination", [("real", "real")]),
    ]
    colors = {"real": "#1f77b4", "shuffled": "#2ca02c", "randinit": "#ff7f0e"}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    bins = np.linspace(0.0, 1.0, 13)
    for ax, (battery, arms) in zip(axs, panels, strict=True):
        for arm, label in arms:
            vals = scores(arm, battery)
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.0,
                color=colors[arm],
                label=f"{label} (n={len(vals)})",
            )
            ax.axvline(float(vals.mean()), color=colors[arm], linestyle="--", linewidth=1.2)
        if battery == "discrimination":
            ax.axvline(0.25, color="0.4", linestyle=":", linewidth=1.2)
            ax.axvline(0.50, color="0.1", linestyle="-.", linewidth=1.2)
        else:
            ax.axvline(0.70, color="0.1", linestyle="-.", linewidth=1.2)
        ax.set_xlabel(f"{battery} score per feature")
        ax.legend(fontsize=8, loc="upper left")
    axs[0].set_ylabel("density of features")
    fig.suptitle("issue1773 per-feature validation score distributions (dashes: arm means)")
    savefig_paper(fig, "battery_score_distributions", dir=FIGDIR)
    plt.close(fig)


def plot_kappa_vs_prevalence() -> None:
    import matplotlib.pyplot as plt

    kappa = json.loads((LABELS / "kappa_report.json").read_text())["axes"]
    unresolved: dict[str, float] = {}
    counts: dict[str, Counter] = {ax: Counter() for ax in AXES}
    n_rows = 0
    with TABLE.open() as fh:
        for line in fh:
            row = json.loads(line)
            n_rows += 1
            axis_labels = row.get("axis_labels") or {}
            for ax_name in AXES:
                counts[ax_name][axis_labels.get(ax_name)] += 1
    for ax_name in AXES:
        unresolved[ax_name] = counts[ax_name]["unresolved"] / n_rows

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax_name in AXES:
        info = kappa[ax_name]
        modal = max(info["prevalence"].values())
        ax1.scatter(modal, info["kappa"], s=60, color="#1f77b4", zorder=3)
        ax1.text(modal, info["kappa"] + 0.015, ax_name, fontsize=8, ha="center")
    ax1.axhline(0.6, color="0.1", linestyle="-.", linewidth=1.2)
    ax1.set_xlabel("modal-label prevalence (fraction of items)")
    ax1.set_ylabel("inter-draw Fleiss kappa")
    ax1.set_ylim(0.0, 0.85)

    xs = np.arange(len(AXES))
    ax2.bar(xs, [unresolved[a] for a in AXES], color="#d62728")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(AXES, rotation=20, ha="right")
    ax2.set_ylabel("majority-vote unresolved rate")
    for x, ax_name in zip(xs, AXES, strict=True):
        ax2.text(
            x, unresolved[ax_name] + 0.002, f"{unresolved[ax_name]:.3f}", ha="center", fontsize=8
        )
    fig.suptitle("issue1773 axis-label reliability: kappa vs prevalence skew; unresolved rate")
    savefig_paper(fig, "kappa_vs_prevalence", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plot_battery_distributions()
    plot_kappa_vs_prevalence()
    print(f"figures written under {FIGDIR}")


if __name__ == "__main__":
    main()
