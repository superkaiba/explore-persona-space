"""Issue #2379 clean-result-critic fix round — per-unit companion figures.

Reads the committed ``eval_results/issue_2379/correlations.json`` (no model
calls, no new data) and renders the two per-unit views the clean-result-critic
round required (Lens 15 / SPEC "Low-level data plot behind every aggregate"):

- ``fig10_perunit_pinned_reads`` — per-condition Spearman rho at the pinned
  layer for the three Result-3 reads (context state / mapped prediction /
  actual-answer ceiling, all at the training-answer reference), one labeled
  line per condition. Per-unit companion to ``fig3_layer_curves``.
- ``fig11_perunit_exploratory_arms`` — per-condition Spearman rho at the
  pinned layer for every exploratory arm in ``fig6_exploratory_arms``
  (mean bars re-drawn light, per-condition points overlaid, condition legend).
  Per-unit companion to ``fig6_exploratory_arms``.

Label/color conventions copied from ``scripts/issue2379_analysis.py`` so the
companion figures read consistently with the aggregate ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless VM; set before any pyplot figure is created
import matplotlib.pyplot as plt
import numpy as np

# Copied verbatim from scripts/issue2379_analysis.py (FAMILY_LABELS subset,
# SETTING_LABELS, COND_LABELS) so both figure families share one vocabulary.
FAMILY_LABELS = {
    "ctx_trainref": "Context state (training-answer ref.)",
    "ans_trainref_mapI": "Predicted answer (training-answer ref.)",
    "ans_sameq_mapI": "Predicted answer (inoc-prompt ref.)",
    "ans_trainref_mapB": "Predicted answer, base map (training-answer ref.)",
    "ans_sameq_mapB": "Predicted answer, base map (inoc-prompt ref.)",
    "ans_trainref_mapI_centered": "Predicted answer, mean-centered (training-answer ref.)",
    "identbias_trainref": "Identity+bias answer (training-answer ref.)",
    "identbias_sameq": "Identity+bias answer (inoc-prompt ref.)",
    "ceiling_trainref": "Actual answer, ceiling (training-answer ref.)",
    "ceiling_sameq": "Actual answer, ceiling (inoc-prompt ref.)",
    "trait_proj_mapI": "Predicted answer, trait projection",
    "tfidf_cos": "TF-IDF text",
    "jaccard": "Jaccard text",
    "seqmatcher": "Sequence-match text",
}
SETTING_LABELS = {"em": "Misalignment", "caps": "Capitalization"}
COND_LABELS = {
    "em_bad_legal_advice": "Bad legal advice",
    "em_bad_medical_advice": "Bad medical advice",
    "em_bad_security_advice": "Bad security advice",
    "em_turner_extreme_sports": "Extreme sports advice",
    "em_turner_risky_financial": "Risky financial advice",
    "caps_french": "French capitalization",
    "caps_german": "German capitalization",
    "caps_spanish": "Spanish capitalization",
}

# fig3's three reads (Result 3) and fig6's exploratory roster (Result 4),
# in the same order the aggregate figures draw them.
FIG3_ARMS = ["ctx_trainref", "ans_trainref_mapI", "ceiling_trainref"]
FIG6_ARMS = [
    "ans_trainref_mapI_centered",
    "ans_trainref_mapB",
    "ans_sameq_mapI",
    "ans_sameq_mapB",
    "ceiling_sameq",
    "identbias_trainref",
    "identbias_sameq",
    "trait_proj_mapI",
    "tfidf_cos",
    "jaccard",
    "seqmatcher",
]
TEXT_ARMS = {"tfidf_cos", "jaccard", "seqmatcher"}


def _pin_value(cond: dict, arm: str, pin: int) -> float | None:
    """Spearman rho for ``arm`` at the pinned layer (text arms are layer-free)."""
    if arm in TEXT_ARMS:
        v = cond["text_families"].get(arm, {}).get("spearman")
    else:
        curve = cond["curves"].get(arm)
        v = None if curve is None else curve["spearman"][pin]
    return None if v is None else float(v)


def _cond_palette(names: list[str]) -> dict[str, str]:
    from explore_persona_space.analysis.paper_plots import paper_palette

    pal = paper_palette(max(len(names), 3))
    return {c: pal[i % len(pal)] for i, c in enumerate(names)}


def _save(fig, stem: str, figdir: Path) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    savefig_paper(fig, stem, dir=figdir)
    plt.close(fig)


def fig_perunit_pinned_reads(data: dict, figdir: Path) -> None:
    """Per-condition slopegraph behind fig3's pinned-layer aggregate reads."""
    pins = data["pins"]
    conds = data["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    for ax, setting in zip(axes, ("em", "caps")):
        names = sorted(c for c, v in conds.items() if v["setting"] == setting)
        pin = pins[setting]
        colors = _cond_palette(names)
        xs = np.arange(len(FIG3_ARMS))
        end_ys: list[tuple[str, float]] = []
        for cname in names:
            ys = [_pin_value(conds[cname], a, pin) for a in FIG3_ARMS]
            ax.plot(xs, ys, "-o", color=colors[cname], lw=1.2, ms=4, alpha=0.9)
            end_ys.append((cname, ys[-1]))
        # De-collide the right-edge condition labels (min vertical gap in data units).
        end_ys.sort(key=lambda t: t[1])
        min_gap = 0.06
        label_y = [y for _, y in end_ys]
        for k in range(1, len(label_y)):
            if label_y[k] - label_y[k - 1] < min_gap:
                label_y[k] = label_y[k - 1] + min_gap
        for (cname, _), ly in zip(end_ys, label_y):
            ax.text(
                xs[-1] + 0.06,
                ly,
                COND_LABELS[cname],
                color=colors[cname],
                fontsize=6.5,
                va="center",
            )
        means = [float(np.mean([_pin_value(conds[c], a, pin) for c in names])) for a in FIG3_ARMS]
        ax.plot(xs, means, "--D", color="#333333", lw=1.6, ms=5, label="mean over conditions")
        ax.axhline(0.0, color="#5A5A5A", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            ["Context state", "Mapped prediction", "Actual-answer\nceiling"], fontsize=8
        )
        ax.set_xlim(-0.3, len(FIG3_ARMS) - 0.3 + 1.1)
        ax.set_ylabel("Spearman rho at pinned layer")
        ax.set_title(
            f"{SETTING_LABELS[setting]} — per-condition reads at decoder layer {pin}",
            fontsize=9,
        )
        ax.legend(fontsize=6.5, loc="lower left")
    _save(fig, "fig10_perunit_pinned_reads", figdir)


def fig_perunit_exploratory(data: dict, figdir: Path) -> None:
    """Per-condition points behind fig6's exploratory-arm mean bars."""
    pins = data["pins"]
    conds = data["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
    for ax, setting in zip(axes, ("em", "caps")):
        names = sorted(c for c, v in conds.items() if v["setting"] == setting)
        pin = pins[setting]
        colors = _cond_palette(names)
        arms = [
            a for a in FIG6_ARMS if any(_pin_value(conds[c], a, pin) is not None for c in names)
        ]
        for i, arm in enumerate(arms):
            vals = [(c, _pin_value(conds[c], arm, pin)) for c in names]
            vals = [(c, v) for c, v in vals if v is not None]
            mean = float(np.mean([v for _, v in vals]))
            ax.bar(i, mean, color="#D8D8D8", width=0.7, zorder=1)
            for j, (cname, v) in enumerate(vals):
                jitter = (j - (len(vals) - 1) / 2.0) * 0.11
                ax.scatter(
                    i + jitter,
                    v,
                    s=16,
                    color=colors[cname],
                    zorder=3,
                    label=COND_LABELS[cname] if i == 0 else None,
                )
        ax.axhline(0.0, color="#5A5A5A", lw=0.8)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels([FAMILY_LABELS[a] for a in arms], rotation=40, ha="right", fontsize=6)
        ax.set_ylabel("Spearman rho at pinned layer")
        ax.set_title(
            f"{SETTING_LABELS[setting]} — per-condition values behind the arm means",
            fontsize=9,
        )
        ax.legend(fontsize=6, loc="lower right", ncol=1)
    _save(fig, "fig11_perunit_exploratory_arms", figdir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--root", type=Path, default=default_root)
    args = parser.parse_args()

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    data = json.loads(
        (args.root / "eval_results/issue_2379/correlations.json").read_text(encoding="utf-8")
    )
    figdir = args.root / "figures/issue_2379"
    fig_perunit_pinned_reads(data, figdir)
    fig_perunit_exploratory(data, figdir)
    print("wrote fig10_perunit_pinned_reads + fig11_perunit_exploratory_arms to", figdir)


if __name__ == "__main__":
    main()
