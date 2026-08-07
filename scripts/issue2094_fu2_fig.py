"""Fold figures for issue #2094 follow-up round `fu2_span_slots`.

Two figures over the committed fu2 artifacts (never recomputed statistics):

1. ``fu2_verdict_composition`` — per-span verdict composition of the family
   reads (clean-separating / separating-but-cap-compromised / separating with
   under 5 pairs / measured-not-separated / not comparable) for the three fu2
   spans, beside the parent grid's fu2-comparable subset (template-inclusive
   query span + context-end, Type-A joint variants). Fractions of each span's
   family reads; counts asserted against `fu2_summary.json` totals.
2. ``fu2_coherence_by_dose`` — judge-incoherent fraction per span x dose x arm
   (pooled over the two layer variants), recomputed from the committed per-row
   tables `fu2_cells.jsonl` / `fu2_null_cells.jsonl`, with 95 percent Wald
   intervals. The mechanism behind the `not_comparable` mass in figure 1.

Writes figures/issue_2094/fu2_verdict_composition.{png,pdf,meta.json} and
figures/issue_2094/fu2_coherence_by_dose.{png,pdf,meta.json}.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    proportion_ci,
    savefig_paper,
    set_paper_style,
)

FU2_DIR = Path("eval_results/issue_2094/f_metrics/fu2")

# Fixed verdict-class order + colorblind-safe (Wong-derived) colors; the same
# class means the same color in every bar (paper-plots section 3.6).
VERDICT_ORDER = [
    "clean_separating",
    "separating_compromised",
    "separating_lt5_pairs",
    "not_separating",
    "not_comparable",
]
VERDICT_LABEL = {
    "clean_separating": "clean separation (steered above null, cap-clean)",
    "separating_compromised": "separating, but cap-hit-compromised",
    "separating_lt5_pairs": "separating, under 5 usable pairs",
    "not_separating": "measured, not separated",
    "not_comparable": "not comparable (coherence / pair floor)",
}
VERDICT_COLOR = {
    "clean_separating": "#009E73",
    "separating_compromised": "#E69F00",
    "separating_lt5_pairs": "#F0E442",
    "not_separating": "#BBBBBB",
    "not_comparable": "#5D5D5D",
}

# Row order (top to bottom): query-side spans first, then the prefix spans.
ROW_ORDER = ["qtext", "qspan", "ce", "pspan_text", "pspan_tmpl"]
ROW_LABEL = {
    "qtext": "query text tokens only\n(this round)",
    "qspan": "query span with template tokens\n(parent grid)",
    "ce": "context-end single position\n(parent grid)",
    "pspan_text": "prefix content tokens only\n(this round)",
    "pspan_tmpl": "whole prefix with template tokens\n(this round)",
}

SLOT_COLOR = {"qtext": "#0072B2", "pspan_text": "#009E73", "pspan_tmpl": "#D55E00"}
SLOT_LABEL = {
    "qtext": "query text tokens only",
    "pspan_text": "prefix content tokens only",
    "pspan_tmpl": "whole prefix with template tokens",
}
DOSE_ORDER = ["a0.5", "a1", "a2", "a4", "replace"]
DOSE_LABEL = {"a0.5": "0.5x", "a1": "1x", "a2": "2x", "a4": "4x", "replace": "full-state\npatch"}


def load_summary() -> dict:
    """Load the committed fu2 verdict summary and sanity-check its totals."""
    summary = json.loads((FU2_DIR / "fu2_summary.json").read_text())
    per_slot = summary["per_slot"]
    assert sorted(per_slot) == ["pspan_text", "pspan_tmpl", "qtext"], sorted(per_slot)
    assert per_slot["qtext"]["n_family_reads"] == 70, per_slot["qtext"]
    assert per_slot["pspan_tmpl"]["n_family_reads"] == 50
    assert per_slot["pspan_text"]["n_family_reads"] == 50
    parent = summary["parent_comparables"]["per_slot"]
    assert sorted(parent) == ["ce", "qspan"], sorted(parent)
    return summary


def verdict_composition_fig(summary: dict) -> None:
    """Stacked horizontal fraction bars of verdict classes per span."""
    rows = dict(summary["per_slot"])
    rows.update(summary["parent_comparables"]["per_slot"])

    fig, ax = plt.subplots(figsize=(8.4, 4.4), layout="constrained")
    ys = np.arange(len(ROW_ORDER))[::-1]
    for slot, y in zip(ROW_ORDER, ys):
        counts = rows[slot]["verdict_counts"]
        total = rows[slot]["n_family_reads"]
        assert sum(counts.values()) == total, (slot, counts, total)
        left = 0.0
        for verdict in VERDICT_ORDER:
            frac = counts.get(verdict, 0) / total
            ax.barh(
                y,
                frac,
                left=left,
                height=0.62,
                color=VERDICT_COLOR[verdict],
                label=VERDICT_LABEL[verdict] if slot == ROW_ORDER[0] else None,
            )
            left += frac
        assert abs(left - 1.0) < 1e-9, (slot, left)

    ax.set_yticks(ys)
    ax.set_yticklabels([ROW_LABEL[s] for s in ROW_ORDER], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of family reads (setting x layer variant x dose x metric)")
    ax.set_title(
        "verdict composition per intervention span: follow-up spans vs parent-grid comparables",
        loc="left",
        fontsize=10,
    )
    fig.legend(fontsize=7.5, loc="outside lower center", ncols=2)
    savefig_paper(fig, "issue_2094/fu2_verdict_composition", dir="figures/")
    plt.close(fig)
    print("[fu2-fig] wrote figures/issue_2094/fu2_verdict_composition.{png,pdf,meta.json}")


def incoherence_by_dose() -> dict[tuple[str, str, str], tuple[int, int]]:
    """(slot, dose, arm) -> (n_incoherent, n_rows), pooled over layer variants."""
    agg: dict[tuple[str, str, str], list[int]] = collections.defaultdict(lambda: [0, 0])
    for path, arm in (
        (FU2_DIR / "fu2_cells.jsonl", "steered"),
        (FU2_DIR / "fu2_null_cells.jsonl", "null"),
    ):
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                assert r["arm"] == arm, (path.name, r["arm"])
                key = (r["slot"], r["dose"], arm)
                agg[key][0] += int(not r["coherent"])
                agg[key][1] += 1
    out = {k: (v[0], v[1]) for k, v in agg.items()}
    # Grid arithmetic: qtext pools 60 pairs x 2 variants, pspan slots 30 x 2.
    for (slot, _dose, _arm), (_bad, n) in out.items():
        assert n == (120 if slot == "qtext" else 60), (slot, n)
    return out


def coherence_fig() -> None:
    """Judge-incoherent fraction vs dose per span, steered vs shuffled-donor null."""
    agg = incoherence_by_dose()
    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    x = np.arange(len(DOSE_ORDER))
    for slot in ("qtext", "pspan_text", "pspan_tmpl"):
        for arm, ls, filled in (("steered", "-", True), ("null", "--", False)):
            fracs, los, his = [], [], []
            for dose in DOSE_ORDER:
                bad, n = agg[(slot, dose, arm)]
                p = bad / n
                lo, hi = proportion_ci(p, n)
                fracs.append(p)
                los.append(p - lo)
                his.append(hi - p)
            ax.errorbar(
                x + (0.0 if arm == "steered" else 0.06),
                fracs,
                yerr=[los, his],
                fmt="o" + ls,
                color=SLOT_COLOR[slot],
                mfc=SLOT_COLOR[slot] if filled else "white",
                mec=SLOT_COLOR[slot],
                markeredgewidth=1.2,
                markersize=5,
                elinewidth=1.1,
                capsize=2,
                linewidth=1.4,
                label=f"{SLOT_LABEL[slot]} - {arm}",
            )
    ax.axhline(0.02, color="grey", linewidth=0.9, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([DOSE_LABEL[d] for d in DOSE_ORDER])
    ax.set_xlabel("edit dose")
    ax.set_ylabel("judge-incoherent fraction of rollouts")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        "generation coherence per span and dose, steered vs shuffled-donor null"
        " (dotted line: 2.0 percent unpatched anchor baseline)",
        loc="left",
        fontsize=9.5,
    )
    fig.legend(fontsize=7.5, loc="outside lower center", ncols=2)
    savefig_paper(fig, "issue_2094/fu2_coherence_by_dose", dir="figures/")
    plt.close(fig)
    print("[fu2-fig] wrote figures/issue_2094/fu2_coherence_by_dose.{png,pdf,meta.json}")


def main() -> int:
    set_paper_style("blog")
    summary = load_summary()
    verdict_composition_fig(summary)
    coherence_fig()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
