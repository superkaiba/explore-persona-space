"""MATS 2026 poster figure 4: "What does it fail to retrieve?" (deterministic answers).

Restores the deterministic-answer CATEGORY BATTERY figure (user redirect
2026-08-20, superseding the metric-ladder version) with SHORT y-tick labels.
Numbers are read VERBATIM from the pinned committed sidecar of the original
deterministic-condition run — git blob
``e2bf520e9c:docs/posters/mats_2026/figures/plot4_failures_data.json`` —
nothing is recomputed. That sidecar's condition: failures under
5-draw-AVERAGED answer targets (original + 4 fresh on-policy draws; the
committed sampling-noise-free condition — no greedy arm exists in #2202),
raw-euclidean retrieval, 1,988 resample-covered held-out rows, pool 9,941;
180 failures, rank-1 accuracy 0.909. Battery = registered #1738 contrast
family, 10,000-draw bootstrap + 10,000 permutations + BH q=0.05 (seed 2202);
attribution via per-row resample classes (percontext_ranks.csv kres_class).

Panel (a): failure-rate difference (group minus rest, pp) per category, blue =
clears BH FDR at q = 0.05, gray = tested, not significant (legend says only
"significant"/"not significant"; the FDR statement lives in the caption).
Panel (b): attribution split of the 180 failures (map error 90.6% / answer
degeneracy 6.1% / ambiguous 3.3%).

Writes ``docs/posters/mats_2026/figures/plot4_failures.{png,pdf,meta.json}``
and restores the sidecar as ``plot4_failures_data.json`` (verbatim pinned blob).
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"
PINNED_BLOB = "e2bf520e9c:docs/posters/mats_2026/figures/plot4_failures_data.json"

# Short poster labels (caption carries any needed expansion).
SHORT_NAME = {
    "language=en": "English",
    "topic=factual_qa": "factual QA",
    "topic=creative_writing": "creative writing",
    "topic=coding": "coding",
    "topic=advice_howto": "advice",
    "topic=chitchat_social": "chit-chat",
    "topic=translation": "translation",
    "topic=summarization_extraction": "summarization",
    "topic=roleplay_persona": "roleplay",
    "topic=math": "math",
    "refusal_adjacent=yes": "refusal-adjacent",
    "answer_is_refusal=yes": "refusal answer",
    "format=code": "code answer",
    "format=list": "list answer",
    "format=prose": "prose answer",
    "depth=2-2": "2 turns",
    "depth=3-4": "3–4 turns",
    "depth=>=5": "5+ turns",
    "corpus=wildchat": "WildChat",
}


def load_pinned_sidecar() -> dict:
    """The deterministic-condition sidecar, verbatim from the pinned git blob."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "show", PINNED_BLOB],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    d = load_pinned_sidecar()
    battery, n_fail = d["battery"], d["n_failures"]
    att = d["attribution_counts"]
    assert sum(att.values()) == n_fail, att

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.8, 3.7), gridspec_kw={"width_ratios": [2.9, 1.0]}
    )

    # (a) all registered contrasts, BH-significant in blue, rest in gray
    rows = sorted(battery, key=lambda r: r["delta"])
    ys = np.arange(len(rows))
    deltas = np.array([r["delta"] for r in rows]) * 100
    elo = np.maximum(0.0, np.array([r["delta"] - r["ci_lo"] for r in rows]) * 100)
    ehi = np.maximum(0.0, np.array([r["ci_hi"] - r["delta"] for r in rows]) * 100)
    colors = [paper_color("instruct") if r["bh_significant"] else paper_color("null") for r in rows]
    ax_a.barh(
        ys,
        deltas,
        xerr=(elo, ehi),
        color=colors,
        height=0.62,
        error_kw={"lw": 0.9, "capsize": 1.8},
    )
    ax_a.axvline(0, color=paper_color("reference"), lw=0.9)
    ax_a.set_yticks(ys, [SHORT_NAME[r["contrast"]] for r in rows])
    ax_a.set_xticks([-10, 0, 10, 20])
    ax_a.set_xlabel("failure rate vs rest (pp)")
    ax_a.bar([], [], color=paper_color("instruct"), label="significant")
    ax_a.bar([], [], color=paper_color("null"), label="not significant")
    ax_a.legend(loc="lower right", handlelength=1.0)

    # (b) resample attribution of the deterministic-target failures
    order = [
        ("MAP_ATTRIBUTABLE", "map error", paper_color("instruct")),
        ("AMBIGUOUS", "ambiguous", "0.85"),
        ("IRREDUCIBLE", "answer degeneracy", paper_color("null")),
    ]
    bottom = 0.0
    for key, lab, colr in order:
        frac = att.get(key, 0) / n_fail * 100.0
        ax_b.bar([0], [frac], bottom=bottom, color=colr, width=0.55, label=lab)
        bottom += frac
    ax_b.set_xticks([])
    ax_b.set_xlim(-0.6, 0.6)
    ax_b.set_ylim(0, 100)
    ax_b.set_ylabel("share of failures (%)")
    ax_b.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), handlelength=1.0)

    savefig_paper(fig, "plot4_failures", dir=OUT_DIR)
    plt.close(fig)

    (OUT_DIR / "plot4_failures_data.json").write_text(json.dumps(d, indent=2))
    print(f"wrote {OUT_DIR}/plot4_failures.{{png,pdf,meta.json}} + restored sidecar")
    for r in sorted(battery, key=lambda r: -r["delta"]):
        print(
            f"  (a) {SHORT_NAME[r['contrast']]}: {r['delta'] * 100:+.2f}pp "
            f"[{r['ci_lo'] * 100:+.2f}, {r['ci_hi'] * 100:+.2f}] "
            f"BH={'YES' if r['bh_significant'] else 'no'}"
        )
    for key, lab, _ in order:
        print(f"  (b) {lab}: {att[key]}/{n_fail} = {att[key] / n_fail * 100:.1f}%")


if __name__ == "__main__":
    main()
