"""MATS 2026 poster figure 4: C3 failure analysis, panels (a)+(b) only.

Poster restyling of the paper figure ``figures/paper/c3_failure_attribution``
(generator: ``scripts/issue2202_regen_figs.py::fig_c3_failure_analysis_iclr``)
with panel (c) — the per-architecture rank-1 accuracy plot — removed on user
order. Reads only committed ``eval_results/issue_2202`` JSONs:

- panel (a): the 13 BH-significant failure-rate contrasts by context category
  (``composition_stats.json .banked_battery``; 10,000-draw bootstrap 95% CIs),
- panel (b): resample attribution of the covered rank-1 failures
  (``attribution.json .classes_over_fail1``).

Writes ``docs/posters/mats_2026/figures/plot4_failures.{png,pdf,meta.json}``.
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "eval_results" / "issue_2202"
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"

# Human-readable prose per contrast key (poster rule: never raw snake_case).
CONTRAST_NAME = {
    "language=en": "English",
    "topic=factual_qa": "factual QA topic",
    "topic=coding": "coding topic",
    "topic=advice_howto": "advice / how-to topic",
    "topic=harmful_or_unsafe_request": "harmful-request topic",
    "topic=roleplay_persona": "roleplay / persona topic",
    "topic=nsfw": "NSFW topic",
    "topic=other": "'other' topic",
    "refusal_adjacent=yes": "refusal-adjacent request",
    "answer_is_refusal=yes": "answer is a refusal",
    "format=code": "code-formatted answer",
    "depth=>=5": "deep conversation (5+ turns)",
    "corpus=wildchat": "WildChat corpus",
}


def main() -> None:
    set_paper_style("iclr", font_scale=1.5)
    comp = json.loads((EV / "composition_stats.json").read_text())
    att = json.loads((EV / "attribution.json").read_text())

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [3.0, 1.0]}
    )

    # (a) BH-significant composition contrasts (failure-rate difference, pp)
    rows = sorted(
        (r for r in comp["banked_battery"] if r["bh_significant"]), key=lambda r: r["delta"]
    )
    ys = np.arange(len(rows))
    deltas = np.array([r["delta"] for r in rows]) * 100
    elo = np.maximum(0.0, np.array([r["delta"] - r["ci_lo"] for r in rows]) * 100)
    ehi = np.maximum(0.0, np.array([r["ci_hi"] - r["delta"] for r in rows]) * 100)
    ax_a.barh(
        ys,
        deltas,
        xerr=(elo, ehi),
        color=paper_color("instruct"),
        height=0.62,
        error_kw={"lw": 1.0, "capsize": 2.0},
    )
    ax_a.axvline(0, color=paper_color("reference"), lw=0.9)
    ax_a.set_yticks(ys, [CONTRAST_NAME[r["contrast"]] for r in rows])
    ax_a.set_xticks([-10, 0, 10, 20])
    ax_a.set_xlabel("failure-rate difference (pp)")

    # (b) resample attribution of the covered rank-1 failures
    counts = att["classes_over_fail1"]
    order = [
        ("MAP_ATTRIBUTABLE", "map error", paper_color("instruct")),
        ("AMBIGUOUS", "ambiguous", "0.85"),
        ("IRREDUCIBLE", "answer degeneracy", paper_color("null")),
    ]
    covered = sum(counts[k] for k, _, _ in order)
    bottom = 0.0
    for key, lab, colr in order:
        frac = counts[key] / covered * 100.0
        ax_b.bar([0], [frac], bottom=bottom, color=colr, width=0.55, label=lab)
        bottom += frac
    ax_b.set_xticks([])
    ax_b.set_xlim(-0.6, 0.6)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("share of covered failures (%)")
    ax_b.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.0)

    savefig_paper(fig, "plot4_failures", dir=OUT_DIR)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/plot4_failures.{{png,pdf,meta.json}}")
    for r in rows:
        print(
            f"  (a) {CONTRAST_NAME[r['contrast']]}: delta={r['delta'] * 100:+.2f}pp "
            f"[{r['ci_lo'] * 100:+.2f}, {r['ci_hi'] * 100:+.2f}]"
        )
    for key, lab, _ in order:
        print(f"  (b) {lab}: {counts[key]:,} / {covered:,} = {counts[key] / covered * 100:.1f}%")


if __name__ == "__main__":
    main()
