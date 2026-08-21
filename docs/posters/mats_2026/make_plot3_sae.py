"""Poster plot 3 — the iterative-partialling (forward-stepwise) results, one panel.

MATS 2026 poster section 3: the abstraction-level story, carried by the
forward-stepwise concordance series itself. Each row is one stepwise round's
WINNER: round 0 conditions on nothing, and every later round adds the previous
winner to the control set (coarsened exact matching), re-stratifies, and
re-scores every remaining candidate — so each row's value is the concordance c
(AUROC scale) of that round's winner AFTER partialling out every row above it.

Rounds 0-1 are the firing-activity controls (the setup, not the finding). Once
they are partialled out, the surviving positive predictor is the HIGH-LEVEL
class (speaker identity / persona, c=0.671), while the low-level / content
classes fall BELOW chance (logit-footprint suppressing 0.351 / promoting 0.358,
topic 0.374). The matryoshka COARSEST-tier read is drawn as a detached
reference row (diamond): it is structurally EXCLUDED from the stepwise
selection (different dictionary/layer/panel), so its value is the
single-control activity-matched concordance — direction-only, disclosed.

Two display drops, both disclosed and neither changing a drawn value:
- Round 0 (the second activity control, "Fires on BOTH context and answer
  side", c=0.768) is NOT drawn but REMAINS in the control set every drawn row
  conditions on. Only one activity control is shown, in gray. The poster
  caption says so; the data JSON carries the round in full.
- Rounds 6-13 are dropped outright (kept in the data JSON): |c-0.5| <= 0.07 and
  shrinking toward the series' 0.02 stop resolution, with per-control bins
  coarsening 6 -> 2 and retained-pair fraction falling to 0.019 — the
  artifact's own caveats mark them the least reliable rounds. Nothing below
  them is drawn, so no drawn row's conditioning statement is affected.

Sources (nothing hand-typed):
- figures/issue_1482/concordance/writeup_stepwise.meta.json
  (scripts/issue1482_concordance_stepwise.py; layer-19 BatchTopK k=64
  dictionary, 131,072-wide, n=120,716 features; R^2 target = full-width
  dense-context -> SAE ridge map ridge__mean,
  data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy)
- figures/issue_1482/concordance/writeup_concordance.meta.json
  (the matryoshka activity-matched reference row; layer-20 SAELens matryoshka
  lmsys dictionary, 16,384-feature panel)

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot3_sae.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
STEPWISE = REPO / "figures/issue_1482/concordance/writeup_stepwise.meta.json"
MATCHED = REPO / "figures/issue_1482/concordance/writeup_concordance.meta.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

ROUNDS_SHOWN = (1, 2, 3, 4, 5)  # drawn, top -> bottom, in stepwise round order
CONTROL_ROUNDS = (1,)  # drawn gray: the displayed firing-activity control
# Round 0 is a SECOND activity control ("Fires on BOTH context and answer side",
# c=0.768). It is dropped from the panel for legibility but is STILL part of the
# control set every drawn row conditions on — disclosed in the caption and here,
# never silently. Rounds 6-13 are dropped outright (nothing below them is drawn,
# so no conditioning statement is affected).
ROUNDS_HIDDEN_BUT_CONDITIONED = (0,)
MATRYOSHKA_NAME = "Matryoshka dictionary tier (coarser = earlier tier)"

# stepwise winner name → short poster tick label (caption carries expansions).
# The two logit-footprint rows are the Gurnee footprint classes: high kurtosis
# with positive skew = the feature pushes a SMALL number of vocabulary tokens up
# ("promoting"), negative skew = pushes a small number down ("suppressing")
# — scripts/issue1482_continuous_predictors.py::classify_promoting.
SHORT_LABELS = {
    "Mean activation (over all answers)": "mean activation",
    "Speaker: identity / disposition": "related to speaker identity",
    "Logit footprint: suppressing": "suppresses a few tokens",
    "Logit footprint: promoting": "boosts a few tokens",
    "Content type: topic": "related to topic",
    MATRYOSHKA_NAME: "coarsest tier",
}


def load_rounds() -> list[dict]:
    """All 14 stepwise rounds (winner + conditional c + conditioning diagnostics)."""
    with open(STEPWISE) as f:
        d = json.load(f)
    return [
        {
            "round": r["round"],
            "winner": r["winner"],
            "c": r["winner_c"],
            "controls": r["controls"],
            "bins": r["bins"],
            "pair_frac": r["pair_frac"],
            "n": r["n"],
        }
        for r in d["rounds"]
    ]


def load_matryoshka() -> dict:
    """The activity-matched matryoshka general-tier reference row."""
    with open(MATCHED) as f:
        d = json.load(f)
    (row,) = [r for r in d["values"] if r["name"] == MATRYOSHKA_NAME]
    return {
        "name": row["name"],
        "matched": row["matched"],
        "pooled": row["pooled"],
        "arm": row.get("arm"),
        "n_pos": row.get("n_pos"),
    }


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    rounds = load_rounds()
    by_round = {r["round"]: r for r in rounds}
    kept = [by_round[i] for i in ROUNDS_SHOWN]
    n_shown = len(kept)
    matry = load_matryoshka()

    c_ctrl = paper_color("null")
    c_win = paper_color("instruct")
    c_matry = paper_color("identity_bias")

    # wider canvas than the other panels: the tick labels are long sentences, so
    # at poster font size a 6.8in canvas left the data region narrow enough that
    # consecutive x tick labels ran into each other
    fig, ax = plt.subplots(figsize=(8.6, 2.6), constrained_layout=True)

    # stepwise rounds top -> bottom in round order; matryoshka detached below
    ys = [float(n_shown) + 0.5 - i for i in range(n_shown)]
    y_matry = 0.0

    ax.axvline(0.5, color=c_ctrl, lw=0.8, ls="--", zorder=1)
    for yi, r in zip(ys, kept):
        is_ctrl = r["round"] in CONTROL_ROUNDS
        color = c_ctrl if is_ctrl else c_win
        ax.hlines(yi, 0.5, r["c"], color=color, lw=1.4, zorder=2)
    ctrl = [(yi, r) for yi, r in zip(ys, kept) if r["round"] in CONTROL_ROUNDS]
    wins = [(yi, r) for yi, r in zip(ys, kept) if r["round"] not in CONTROL_ROUNDS]
    ax.scatter(
        [r["c"] for _, r in ctrl],
        [yi for yi, _ in ctrl],
        s=30,
        color=c_ctrl,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        label="activity control",
    )
    ax.scatter(
        [r["c"] for _, r in wins],
        [yi for yi, _ in wins],
        s=30,
        color=c_win,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        label="winner after rows above",
    )
    ax.hlines(y_matry, 0.5, matry["matched"], color=c_matry, lw=1.4, zorder=2)
    ax.scatter(
        [matry["matched"]],
        [y_matry],
        s=38,
        marker="D",
        color=c_matry,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        label="coarsest tier (activity-matched)",
    )

    ax.set_yticks([*ys, y_matry])
    ax.set_yticklabels([SHORT_LABELS[r["winner"]] for r in kept] + [SHORT_LABELS[MATRYOSHKA_NAME]])
    ax.set_ylim(-0.7, n_shown + 1.1)
    # at poster font size the parenthetical overran the canvas, and a lower-left
    # legend sat on top of the bottom two tick labels; both move out of the axes
    ax.set_xlabel(r"concordance with per-feature $R^2$")
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncols=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
        fontsize="small",
    )

    paths = savefig_paper(fig, "plot3_sae_predictors", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")

    data = {
        "sources": {
            "stepwise": str(STEPWISE.relative_to(REPO)),
            "matryoshka_reference": str(MATCHED.relative_to(REPO)),
        },
        "statistic": (
            "Forward-stepwise concordance (Harrell c, AUROC scale) between each round's "
            "winning predictor and per-feature held-out R^2. Round 0 conditions on "
            "nothing; each later round adds the previous winner to the control set via "
            "coarsened exact matching and re-scores the remaining candidates, so each "
            "row is conditional on every row above it. 0.5 = chance. NO confidence "
            "intervals exist anywhere in this analysis; differences under ~0.02 should "
            "not be read (the series' own stop resolution)."
        ),
        "battery_target": (
            "per-feature held-out R^2 of the full-width dense-context -> SAE-feature "
            "ridge map (ridge__mean, layer-19 BatchTopK k=64 dictionary, 131,072 "
            "features, n=120,716 with finite R^2 + covariates; splits 120,000 train / "
            "2,000 val / 20,000 holdout answers, #1482 single-turn corpus)"
        ),
        "rounds_plotted": kept,
        "rounds_hidden_but_conditioned": [by_round[i] for i in ROUNDS_HIDDEN_BUT_CONDITIONED],
        "rounds_hidden_but_conditioned_reason": (
            "Round 0 ('Fires on BOTH context and answer side', c=0.768) is a SECOND "
            "firing-activity control. It is not drawn — the panel shows one activity "
            "control for legibility — but it REMAINS in the control set that every "
            "drawn row conditions on, so the drawn values are unchanged from the full "
            "series. The poster caption states this; it is never an undisclosed drop."
        ),
        "rounds_dropped": [by_round[i] for i in sorted(by_round) if i > max(ROUNDS_SHOWN)],
        "rounds_dropped_reason": (
            "rounds 6-13: |c-0.5| <= 0.07 shrinking toward the 0.02 stop resolution, "
            "per-control bins coarsen 6 -> 2, retained-pair fraction falls to 0.019 — "
            "the artifact's own caveats mark these the least reliable rounds; dropped "
            "for poster legibility, kept here in full. Nothing below them is drawn, so "
            "no drawn row's conditioning statement is affected."
        ),
        "matryoshka_reference": {
            **matry,
            "note": (
                "NOT part of the stepwise selection — scored on a DIFFERENT dictionary "
                "(layer-20 SAELens matryoshka jumprelu k=100 lmsys, 16,384-feature "
                "tier-stratified panel, 24k fit / 6k score answers), so it has no value "
                "to stratify the layer-19 battery on. Drawn as a detached reference "
                "row: concordance of COARSER (more general) tier with R^2, pairs "
                "matched within activity deciles (single control). Direction-only; "
                "never magnitude-comparable to the battery rows."
            ),
        },
    }
    out_json = OUT_DIR / "plot3_sae_predictors_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1)
    print(f"data: {out_json}")


if __name__ == "__main__":
    main()
