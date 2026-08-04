"""Issue #1482 context-extremes round — mean per-context R^2 per judged category, by arm.

D4 of the "What contexts is the mapping bad at predicting?" writeup section: a
grouped bar plot of the mean per-context R^2 (1 - nerr, nerr = ||v_hat-v||^2 /
||v-mu_eval||^2, the #1482 convention) per judged topic category, grouped by
category and coloured by arm (context vector / prefix end state / query only).

Corpus: #1738 MULTI-TURN holdout, n=9,941, L19 ridge; all three arms score the
SAME target (the answer state generated under the full context — verified
bitwise, see twoway_residual.json design). Inputs are banked:
  - eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv
    (per-context nerr for all three arms + judged labels, ci-keyed)
  - eval_results/issue_1738/judge_labels/labels.json (instrument provenance)

95% CIs are a paired nonparametric bootstrap over contexts WITHIN a category
(shared index draws across arms, B=2000, seed pinned) rendered as non-negative
offsets. 0 GPU; CPU-only, banked artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

CSV = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1738"
    / "bare_query"
    / ("percontext_summary_L19_ridge.csv")
)
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1482"
OUT_JSON = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "context_extremes" / "category_r2_by_arm.json"
)
STEM = "context_category_by_arm"
SEED = 1482
N_BOOT = 2000
# (csv column, legend label) — the three input states, result1-figure naming.
ARMS = (
    ("nerr_context_L19_ridge", "Context vector"),
    ("nerr_prefix_L19_ridge", "Prefix end state"),
    ("nerr_bare_L19_ridge", "Query only"),
)


def main() -> None:
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_unlabeled = sum(1 for r in rows if not r["topic"])
    labeled = [r for r in rows if r["topic"]]
    print(f"[cat] rows={len(rows)} labeled={len(labeled)} unlabeled={n_unlabeled}")

    topics = sorted({r["topic"] for r in labeled})
    rng = np.random.default_rng(SEED)
    stats: dict[str, dict] = {}
    for topic in topics:
        sub = [r for r in labeled if r["topic"] == topic]
        n = len(sub)
        idx = rng.integers(0, n, size=(N_BOOT, n))  # shared draws -> paired across arms
        stats[topic] = {"n": n, "arms": {}}
        for col, label in ARMS:
            r2 = 1.0 - np.array([float(r[col]) for r in sub])
            boots = r2[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            stats[topic]["arms"][label] = {
                "mean_percontext_r2": float(r2.mean()),
                "ci95": [float(lo), float(hi)],
            }

    # order categories by the context-arm mean, best-predicted first
    order = sorted(topics, key=lambda t: -stats[t]["arms"]["Context vector"]["mean_percontext_r2"])

    set_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    colors = paper_palette(3)
    width = 0.26
    x = np.arange(len(order))
    for ai, (_col, label) in enumerate(ARMS):
        means = np.array([stats[t]["arms"][label]["mean_percontext_r2"] for t in order])
        lo = np.array([stats[t]["arms"][label]["ci95"][0] for t in order])
        hi = np.array([stats[t]["arms"][label]["ci95"][1] for t in order])
        yerr = np.vstack([np.maximum(0, means - lo), np.maximum(0, hi - means)])
        ax.bar(
            x + (ai - 1) * width,
            means,
            width=width,
            color=colors[ai],
            label=label,
            yerr=yerr,
            capsize=2,
            error_kw={"lw": 0.9},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean per-context $R^2$  ($1 - $ normalized error)")
    ax.set_title(
        "Held-out prediction quality by judged context category — "
        "#1738 multi-turn holdout (n=9,941), L19 ridge",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.axhline(0.0, color="0.4", lw=0.8)
    fig.tight_layout()
    paths = savefig_paper(fig, STEM, dir=FIG_DIR)

    # augment the savefig meta with sources + n per cell (brief requirement)
    meta_p = FIG_DIR / f"{STEM}.meta.json"
    meta = json.loads(meta_p.read_text())
    meta["sources"] = [
        str(CSV.relative_to(PROJECT_ROOT)),
        "eval_results/issue_1738/judge_labels/labels.json",
    ]
    meta["corpus"] = "#1738 multi-turn holdout, n=9,941, L19 ridge; identical targets across arms"
    meta["metric"] = "mean per-context R^2 = mean(1 - nerr); nerr = ||v_hat-v||^2/||v-mu_eval||^2"
    meta["ci"] = f"paired bootstrap over contexts within category, B={N_BOOT}, seed={SEED}"
    meta["n_per_cell"] = {t: stats[t]["n"] for t in order}
    meta["n_unlabeled_excluded"] = n_unlabeled
    meta_p.write_text(json.dumps(meta, indent=1))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "corpus": meta["corpus"],
                "metric": meta["metric"],
                "ci": meta["ci"],
                "category_order_plotted": order,
                "n_unlabeled_excluded": n_unlabeled,
                "stats": stats,
                "caveat_other": "category 'other' is a heterogeneous catch-all (n="
                + str(stats.get("other", {}).get("n", 0))
                + "); do not headline it",
            },
            indent=1,
        )
    )
    for t in order:
        s = stats[t]
        line = "  ".join(
            f"{label.split()[0]}={s['arms'][label]['mean_percontext_r2']:.3f}" for _c, label in ARMS
        )
        print(f"[cat] {t:28s} n={s['n']:5d}  {line}")
    print(f"[cat] figure: {paths}")
    print(f"[cat] stats json: {OUT_JSON}")


if __name__ == "__main__":
    main()
