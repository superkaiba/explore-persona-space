"""Issue #444: bystander-prior predictor vs geometric predictors of fact leakage.

Panel A: per-persona base-model log P(taught data | persona) (the bystander
prior) vs taught-fact leak rate, one series per contrastive-negative recipe.
Higher base prior -> more leak is the predicted (positive) direction.

Panel B: pooled rank-correlation of each candidate predictor with leak across
the 3 variance recipes (n=18). The bystander prior is POSITIVE; persona-vector
cosine and output-distribution JS (on-topic) are NEGATIVE -- they rank the most-
leaky content-fit persona (local_historian) as the most distant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

BASE = json.loads(
    (REPO / "eval_results/issue_444/bystander_logprob/logprob_results.json").read_text()
)["summary"]
LEAK = json.loads(
    (REPO / "eval_results/issue_444/bystander_logprob/leak_rates_snapshot.json").read_text()
)["recipes"]
DIST = json.loads((REPO / "eval_results/issue_444/persona_distance_topic/results.json").read_text())

NT = [
    "local_historian",
    "local_resident",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
]
LABEL = {
    "local_historian": "local historian",
    "local_resident": "local resident",
    "assistant": "assistant",
    "software_engineer": "SWE",
    "kindergarten_teacher": "teacher",
    "no_system": "no system",
}
base = {p: BASE[p]["mean_logprob_per_tok"] for p in NT}
cos_on = {p: DIST["cosine"]["on_topic"][p]["21"] for p in NT}
js_on = {p: DIST["js_similarity"]["on_topic"][p] for p in NT}
var_recipes = [r for r in LEAK if len({LEAK[r][p] for p in NT}) > 1]
RECIPE_LABEL = {
    "hand-written-contradictory-cn": "contradictory neg.",
    "hand-written-suppression-cn": "refusal neg.",
    "on-policy-suppression-cn": "on-policy neg.",
}


def pooled(pred: dict[str, float]) -> tuple[float, float, float]:
    X, L = [], []
    for r in var_recipes:
        for p in NT:
            X.append(pred[p])
            L.append(LEAK[r][p])
    rho, pv = spearmanr(X, L)
    rp, _ = pearsonr(X, L)
    return rho, pv, rp


def main() -> None:
    set_paper_style()
    pal = paper_palette(4)
    colors = [pal[i] for i in range(len(var_recipes))]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: base prior vs leak, per recipe
    for r, c in zip(var_recipes, colors, strict=True):
        xs = [base[p] for p in NT]
        ys = [LEAK[r][p] for p in NT]
        axA.scatter(xs, ys, color=c, s=55, label=RECIPE_LABEL.get(r, r), zorder=3)
    # label personas once (at on-policy recipe y)
    rlab = "on-policy-suppression-cn"
    for p in NT:
        axA.annotate(
            LABEL[p],
            (base[p], LEAK[rlab][p]),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axA.set_xlabel(
        "base-model log P(taught data | persona)  (per-token nats; higher = stronger prior)"
    )
    axA.set_ylabel("taught-fact leak rate (A-family)")
    rho = pooled(base)[0]
    axA.set_title(f"A. Bystander prior vs leak (Spearman {rho:+.2f})")
    axA.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel B: predictor comparison (pooled Spearman)
    preds = {
        "bystander\nprior": pooled(base)[0],
        "cosine\n(on-topic)": pooled(cos_on)[0],
        "JS\n(on-topic)": pooled(js_on)[0],
    }
    names = list(preds)
    vals = [preds[n] for n in names]
    bar_colors = [pal[2] if v > 0 else pal[3] for v in vals]
    axB.bar(names, vals, color=bar_colors, zorder=3)
    axB.axhline(0, color="0.4", lw=0.8)
    for i, v in enumerate(vals):
        axB.annotate(f"{v:+.2f}", (i, v), ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    axB.set_ylabel("pooled Spearman rho with leak (n=18)")
    axB.set_title("B. Bystander prior flips the sign vs geometry")
    axB.set_ylim(-0.7, 0.5)

    fig.suptitle(
        "#444 fact leakage: eval persona's base prior beats (reverses) representational distance",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = REPO / "figures/issue_444/bystander_logprob/bystander_vs_geometry"
    out.parent.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, str(out))
    print("WROTE", out)

    # save correlations json
    corr = {
        "_doc": "Pooled (n=18, 3 variance recipes x 6 non-teach personas) and per-recipe (n=6) "
        "correlations of each predictor with the A-family taught-fact leak rate. Positive = "
        "predicts in the right direction (more prior/closer => more leak).",
        "pooled": {
            "bystander_logprob": dict(
                zip(["spearman", "spearman_p", "pearson"], pooled(base), strict=True)
            ),
            "cosine_on_topic_L21": dict(
                zip(["spearman", "spearman_p", "pearson"], pooled(cos_on), strict=True)
            ),
            "js_on_topic": dict(
                zip(["spearman", "spearman_p", "pearson"], pooled(js_on), strict=True)
            ),
        },
        "per_recipe_spearman_bystander": {
            r: float(spearmanr([base[p] for p in NT], [LEAK[r][p] for p in NT])[0])
            for r in var_recipes
        },
        "per_persona": {
            p: {
                "base_logprob": base[p],
                "cosine_on": cos_on[p],
                "js_on": js_on[p],
                **{f"leak_{RECIPE_LABEL.get(r, r)}": LEAK[r][p] for r in var_recipes},
            }
            for p in NT
        },
    }
    cpath = REPO / "eval_results/issue_444/bystander_logprob/correlations.json"
    cpath.write_text(json.dumps(corr, indent=2, default=float))
    print("WROTE", cpath)


if __name__ == "__main__":
    main()
