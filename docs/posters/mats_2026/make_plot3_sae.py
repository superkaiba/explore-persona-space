"""Poster plot 3 — which SAE-feature classes the context->feature map predicts,
after partialling out firing activity.

ONE compact single-panel lollipop figure for the MATS 2026 poster (section 3):
the abstraction-level story. Rows: every judged label class from the #1482
full-width predictor battery, PLUS the matryoshka dictionary-tier row (diamond
marker — a different dictionary/panel, disclosed; same-direction claim only).
x = activity-matched concordance c (AUROC scale) between the class indicator
and per-feature held-out R^2: pairs are formed only WITHIN per-token
firing-rate deciles, so the read survives the firing-activity confound
(the single-shot ancestor of the forward-stepwise / iterative-partialling
series, whose round-2 ranking agrees: speaker identity wins, topic and the
logit-footprint classes land below 0.5).

High-level classes (speaker identity / persona; the matryoshka GENERAL tier)
sit far above chance; topic-content features sit far below.

Source of every number (nothing hand-typed):
figures/issue_1482/concordance/writeup_concordance.meta.json
(written by scripts/issue1482_concordance_writeup_figs.py; battery target =
dense-context -> SAE ridge map ridge__mean, full 131,072-feature dictionary,
n = 120,716 features; matryoshka arm = layer-20 SAELens matryoshka lmsys
dictionary, 16,384-feature panel,
eval_results/issue_1482/matryoshka_tier/perfeature_m_lmsys_default.npz).

Run:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python docs/posters/mats_2026/make_plot3_sae.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "figures/issue_1482/concordance/writeup_concordance.meta.json"
OUT_DIR = Path(__file__).resolve().parent / "figures"

MATRYOSHKA_NAME = "Matryoshka dictionary tier (coarser = earlier tier)"

# artifact row name → short poster tick label (caption carries the expansions)
SHORT_LABELS = {
    "Speaker: identity / disposition": "speaker identity",
    MATRYOSHKA_NAME: "matryoshka: general tier",
    "Interpretable (autointerp)": "interpretable",
    "Content type: task format": "task format",
    "Content type: syntax": "syntax",
    "Abstraction: token surface": "token surface",
    "Content type: entity": "entity",
    "Speaker: register / style": "register",
    "Content type: operation": "operation",
    "Speaker: language of the text": "language",
    "Abstraction: abstract contextual": "abstract contextual",
    "Abstraction: lexical semantic": "lexical semantic",
    "Content type: topic": "topic",
}


def load_rows() -> list[dict]:
    """All judged label classes + the matryoshka tier row, sorted by matched c."""
    with open(SOURCE) as f:
        d = json.load(f)
    rows = []
    for r in d["values"]:
        judged = r["family"] == "judged"
        matryoshka = r["name"] == MATRYOSHKA_NAME
        if not (judged or matryoshka):
            continue
        rows.append(
            {
                "name": r["name"],
                "label_short": SHORT_LABELS[r["name"]],
                "matched": r["matched"],
                "pooled": r["pooled"],
                "m_both": r.get("m_both"),
                "n_pos": r.get("n_pos"),
                "arm": r.get("arm", "battery-l19-fullwidth"),
                "is_matryoshka": matryoshka,
            }
        )
    rows.sort(key=lambda r: r["matched"], reverse=True)
    return rows


def main() -> None:
    set_paper_style("iclr")
    rows = load_rows()
    c_battery = paper_color("instruct")
    c_matry = paper_color("identity_bias")
    c_null = paper_color("null")

    fig, ax = plt.subplots(figsize=(5.6, 2.55), constrained_layout=True)
    y = np.arange(len(rows), dtype=float)[::-1]  # best at the top

    ax.axvline(0.5, color=c_null, lw=0.8, ls="--", zorder=1)
    for yi, r in zip(y, rows):
        color = c_matry if r["is_matryoshka"] else c_battery
        ax.hlines(yi, 0.5, r["matched"], color=color, lw=1.3, zorder=2)
    bat = [(yi, r) for yi, r in zip(y, rows) if not r["is_matryoshka"]]
    mat = [(yi, r) for yi, r in zip(y, rows) if r["is_matryoshka"]]
    ax.scatter(
        [r["matched"] for _, r in bat],
        [yi for yi, _ in bat],
        s=26,
        color=c_battery,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        label="judged label class",
    )
    ax.scatter(
        [r["matched"] for _, r in mat],
        [yi for yi, _ in mat],
        s=34,
        marker="D",
        color=c_matry,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
        label="matryoshka tier (L20 dict.)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([r["label_short"] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("activity-matched concordance with per-feature $R^2$")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4)

    paths = savefig_paper(fig, "plot3_sae_predictors", dir=OUT_DIR)
    for fmt, p in paths.items():
        print(f"{fmt}: {p}")

    data = {
        "source": str(SOURCE.relative_to(REPO)),
        "statistic": (
            "Harrell concordance c (AUROC scale) between the class indicator and "
            "per-feature held-out R^2; pairs formed only WITHIN per-token firing-rate "
            "deciles (activity-matched). 0.5 = chance. No confidence intervals exist "
            "for this statistic in the source artifact; differences under ~0.02 "
            "should not be read."
        ),
        "battery_target": (
            "per-feature held-out R^2 of the full-width dense-context -> SAE-feature "
            "ridge map (ridge__mean, layer-19 BatchTopK k=64 dictionary, 131,072 "
            "features, n=120,716 with finite R^2 + covariates; splits 120,000 train / "
            "2,000 val / 20,000 holdout answers, #1482 single-turn corpus; "
            "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy)"
        ),
        "matryoshka_arm": (
            "layer-20 SAELens matryoshka jumprelu k=100 lmsys dictionary, "
            "16,384-feature tier-stratified panel (24k fit / 6k score answers); the row "
            "is concordance of COARSER tier with R^2, activity-decile-matched within "
            "the panel (eval_results/issue_1482/matryoshka_tier/"
            "perfeature_m_lmsys_default.npz). DIFFERENT dictionary/layer/panel: "
            "same-direction claim only, never row-to-row magnitude comparison."
        ),
        "iterative_partialling_agreement": (
            "The forward-stepwise (iterative) concordance series "
            "(figures/issue_1482/concordance/writeup_stepwise.meta.json) conditions on "
            "{fires-both-sides, mean activation} by round 2 and agrees: "
            "'Speaker: identity / disposition' is the round-2 winner (c=0.671); "
            "'Content type: topic' (c=0.374) and the logit-footprint classes "
            "(c~0.35-0.37) land below 0.5. Matryoshka tier is structurally EXCLUDED "
            "from the stepwise series (different dictionary), hence the matched-"
            "concordance consolidation here."
        ),
        "rows": rows,
    }
    out_json = OUT_DIR / "plot3_sae_predictors_data.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1)
    print(f"data: {out_json}")


if __name__ == "__main__":
    main()
