"""Issue #1769 fu1 revision figures (clean-result-critic Must-Fix round).

Renders three figures:

1. ``fig_dose_ladder`` — re-render with a corrected legend label only
   (the prior "Clean window (CJK <10%, α1.5)" label was false for evil at
   α=1.5, whose decode-arm CJK intrusion is 23%). Data identical to the
   committed figure: raw-treatment ``delta_both`` per behavior at
   α ∈ {1, 2, 4} from ``analysis/headline.json`` and at α ∈ {1.5, 3} from
   ``analysis/alpha{1.5,3}_clean_lattice.json``.
2. ``fig_alpha15_per_question`` / ``fig_alpha3_per_question`` — per-question
   low-level companions to the α=1.5 / α=3 three-treatment lattices,
   mirroring the parent round's ``fig_alpha2_per_question`` conventions
   (three panels, four dodged arms per question, gray per-question
   connectors), from ``judge_fu1_mt600/graded_scores.json`` per-item means.

Usage: ``uv run python scripts/issue1769_fu1_revision_figs.py``
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps BEFORE heavy imports (#1144 invariant)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "eval_results" / "issue_1769" / "analysis"
FU1_SCORES = REPO / "eval_results" / "issue_1769" / "judge_fu1_mt600" / "graded_scores.json"
FIG_DIR = REPO / "figures"

TRAITS = [("evil", "Scheming"), ("sycophancy", "Sycophancy"), ("hallucination", "Hallucination")]
LADDER_COLORS = {"evil": "#d62728", "hallucination": "#9467bd", "sycophancy": "#2ca02c"}
ARMS = [
    ("neither", "Neither", -0.3, "#6EBEEB"),
    ("prefill_only", "Prefill only", -0.1, "#E9AC25"),
    ("decode_only", "Decode only", 0.1, "#2586BC"),
    ("both", "Both", 0.3, "#DA7525"),
]
N_QUESTIONS = 20
N_DRAWS = 10


def load_ladder() -> dict[str, dict[float, float]]:
    """Raw-treatment delta_both per trait at all five doses."""
    headline = json.loads((ANALYSIS / "headline.json").read_text())
    out: dict[str, dict[float, float]] = {t: {} for t, _ in TRAITS}
    for trait, _ in TRAITS:
        ladder = headline["per_trait"][trait]["alpha_ladder"]
        for a in ("1", "2", "4"):
            out[trait][float(a)] = ladder[a]["both"]["delta"]
    for a, fname in ((1.5, "alpha1.5_clean_lattice.json"), (3.0, "alpha3_clean_lattice.json")):
        lattice = json.loads((ANALYSIS / fname).read_text())
        for row in lattice["results"]:
            out[row["trait"]][a] = row["raw"]["delta_both"]
    # Guard: match the previously committed figure's line data exactly.
    expected = {
        "evil": [5.849, 61.60729166666666, 86.0274126984127, 72.43763888888888, 31.55075],
        "hallucination": [
            15.098694444444444,
            25.97166666666667,
            49.59872751322751,
            67.44649206349206,
            28.008435185185185,
        ],
        "sycophancy": [
            3.848000000000001,
            11.318999999999997,
            19.752666666666663,
            47.37525925925926,
            32.51049999999999,
        ],
    }
    for trait, vals in expected.items():
        got = [out[trait][a] for a in (1.0, 1.5, 2.0, 3.0, 4.0)]
        assert all(abs(g - e) < 1e-9 for g, e in zip(got, vals)), (trait, got, vals)
    return out


def render_dose_ladder() -> None:
    ladder = load_ladder()
    alphas = [1.0, 1.5, 2.0, 3.0, 4.0]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axvspan(0.85, 2.0, color="#2ca02c", alpha=0.085, label="Interpretable window (α ≤ 2)")
    ax.axvspan(2.5, 4.5, color="#ff7f0e", alpha=0.055, label="CJK-intrusion-affected (α3-4)")
    for trait, label in [
        ("evil", "Scheming"),
        ("hallucination", "Hallucination"),
        ("sycophancy", "Sycophancy"),
    ]:
        ys = [ladder[trait][a] for a in alphas]
        color = LADDER_COLORS[trait]
        ax.plot(alphas, ys, marker="s", markersize=6, color=color, label=label, zorder=3)
        for x, y in zip(alphas, ys):
            ax.annotate(
                f"{y:.1f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color=color,
            )
    ax.set_xlim(0.85, 4.5)
    ax.set_ylim(-24, 104)
    ax.set_xticks(alphas)
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel("Behavior increase Δ_both (score points)")
    ax.set_title(
        "Dose ladder: behavior strength vs. steering coefficient\n"
        "(raw draws; delta from neither-arm baseline, all traits)",
        loc="left",
    )
    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index(x)
        for x in (
            "Scheming",
            "Hallucination",
            "Sycophancy",
            "Interpretable window (α ≤ 2)",
            "CJK-intrusion-affected (α3-4)",
        )
    ]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        labelspacing=0.25,
        borderaxespad=0.2,
        fontsize=8.5,
    )
    savefig_paper(fig, "issue_1769/fig_dose_ladder", dir=str(FIG_DIR))
    plt.close(fig)


def per_question_means(alpha_key: str | None) -> dict[str, dict[str, list[float | None]]]:
    """Mean graded score per (trait, arm, question) over non-null draws."""
    per_item = json.loads(FU1_SCORES.read_text())["per_item"]
    out: dict[str, dict[str, list[float | None]]] = {}
    for trait, _ in TRAITS:
        out[trait] = {}
        for arm, _, _, _ in ARMS:
            means: list[float | None] = []
            for q in range(N_QUESTIONS):
                mid = (
                    f"{trait}/neither/q{q:02d}"
                    if arm == "neither"
                    else (f"{trait}/{arm}/{alpha_key}/q{q:02d}")
                )
                scores = []
                for d in range(N_DRAWS):
                    item = per_item.get(f"{mid}/d{d}")
                    if item is not None and item["graded_score"] is not None:
                        scores.append(item["graded_score"])
                means.append(sum(scores) / len(scores) if scores else None)
            out[trait][arm] = means
    return out


def render_per_question(alpha_key: str, alpha_label: str, stem: str) -> None:
    data = per_question_means(alpha_key)
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (trait, panel) in zip(axes, TRAITS):
        for q in range(N_QUESTIONS):
            xs, ys = [], []
            for arm, _, dx, _ in ARMS:
                y = data[trait][arm][q]
                if y is not None:
                    xs.append(q + dx)
                    ys.append(y)
            ax.plot(xs, ys, color="#BBBBBB", lw=0.7, zorder=1)
        for arm, label, dx, color in ARMS:
            xs = [q + dx for q in range(N_QUESTIONS) if data[trait][arm][q] is not None]
            ys = [y for y in data[trait][arm] if y is not None]
            ax.scatter(xs, ys, s=22, color=color, label=label, zorder=3)
        ax.set_title(panel, loc="left")
        ax.set_xlabel("Question index (0–19)")
    axes[0].set_ylabel("Mean graded score (0–100)")
    fig.suptitle(f"Per-question mean graded score by arm at α={alpha_label} (raw scoring)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    savefig_paper(fig, f"issue_1769/{stem}", dir=str(FIG_DIR))
    plt.close(fig)


def main() -> None:
    render_dose_ladder()
    render_per_question("a1.5", "1.5", "fig_alpha15_per_question")
    render_per_question("a3", "3", "fig_alpha3_per_question")
    print("done")


if __name__ == "__main__":
    main()
