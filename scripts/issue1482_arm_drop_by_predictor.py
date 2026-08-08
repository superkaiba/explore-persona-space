"""#1482: which KINDS of feature lose the most when the map sees only the prefix,
or only the query?

Three arms fit the SAME target (the answer state generated under the FULL
context) from three different inputs, per-feature R^2 banked at
eval_results/issue_1738/sae_twoway/perfeature/sae_{context,prefix,bare}_r2.npy.

Per feature, the DROP is

    d_prefix = R2_context - R2_prefix        d_bare = R2_context - R2_bare

and for each predictor in the #1482 battery we report the concordance of that
DROP on the predictor: c = P(in a random pair the predictor separates, the
feature with the higher predictor value is the one that DROPS MORE). 0.5 =
chance. c > 0.5 means features carrying the property lose MORE when the input is
cut back; c < 0.5 means they are comparatively SPARED.

Same statistic and same pair-restriction convention as the rest of the
concordance battery, so rows are comparable to those figures. Data only -- no
interpretation is drawn here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/scripts")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/matplotlib import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import issue1482_concordance_writeup_figs as WF  # noqa: E402
from issue1482_concordance_fig import concordance  # noqa: E402

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
R2_DIR = REPO / "eval_results/issue_1738/sae_twoway/perfeature"
OUT = REPO / "figures/issue_1482/concordance"
STEM = "writeup_arm_drop_by_predictor"
TOP_N = 22  # rows drawn, ranked by the larger of the two drops


def main() -> None:
    b = WF.battery()
    vecs, n = b["vecs"], b["n"]
    fam = {r["name"]: r["family"] for r in b["rows"]}

    arms = {a: np.load(R2_DIR / f"sae_{a}_r2.npy") for a in ("context", "prefix", "bare")}
    for a, v in arms.items():
        print(f"[drop] {a:8s} n={len(v):,} finite={np.isfinite(v).sum():,}")

    # The battery's own `ok` mask is applied inside battery(); rebuild the same
    # selection here so the drop vectors line up with the predictor vectors.
    ok = b["ok"] if "ok" in b else None
    if ok is None:
        raise SystemExit(
            "battery() does not expose its `ok` mask — cannot align the arm R^2 "
            "vectors with the predictor vectors without it"
        )
    d_prefix = (arms["context"] - arms["prefix"])[ok]
    d_bare = (arms["context"] - arms["bare"])[ok]
    finite = np.isfinite(d_prefix) & np.isfinite(d_bare)
    print(f"[drop] aligned n={ok.sum():,}  finite-both={finite.sum():,}")
    print(f"[drop] mean drop  prefix={np.nanmean(d_prefix):+.4f}  bare={np.nanmean(d_bare):+.4f}")

    strata = [np.flatnonzero(finite)]
    rows = []
    for name, v in vecs.items():
        rows.append(
            {
                "name": name,
                "family": fam.get(name, "?"),
                "c_drop_prefix": concordance(v, d_prefix, strata),
                "c_drop_bare": concordance(v, d_bare, strata),
            }
        )
    rows.sort(key=lambda r: -max(abs(r["c_drop_prefix"] - 0.5), abs(r["c_drop_bare"] - 0.5)))
    shown = rows[:TOP_N]

    print(f"\n{'predictor':52s} {'drop|prefix':>12s} {'drop|bare':>11s}")
    for r in shown:
        print(f"{r['name']:52s} {r['c_drop_prefix']:12.3f} {r['c_drop_bare']:11.3f}")

    fig, ax = plt.subplots(figsize=(9.8, 0.34 * len(shown) + 2.4))
    y = np.arange(len(shown))[::-1]
    for yi, r in zip(y, shown):
        c = WF.FAMILY_COLOR.get(r["family"], "#888888")
        ax.plot([r["c_drop_prefix"], r["c_drop_bare"]], [yi, yi], color=c, lw=1.0, alpha=0.5)
        ax.scatter([r["c_drop_prefix"]], [yi], s=44, marker="o", color=c)
        ax.scatter(
            [r["c_drop_bare"]], [yi], s=52, marker="D", facecolor="white", edgecolor=c, lw=1.4
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in shown], fontsize=8.0)
    for tick, r in zip(ax.get_yticklabels(), shown):
        tick.set_color(WF.FAMILY_COLOR.get(r["family"], "#888888"))
    ax.axvline(0.5, color="0.35", lw=1.0)
    ax.set_xlabel(
        "concordance of the R² DROP on the predictor  —  0.5 = chance\n"
        "> 0.5: features with this property lose MORE when the input is cut back"
    )
    ax.set_title(
        "Which kinds of feature lose most when the map sees only the prefix, or only the query?\n"
        f"drop = R²(full context) − R²(arm), per feature · n = {int(finite.sum()):,} · "
        f"top {len(shown)} predictors by |c − 0.5|",
        fontsize=10.4,
    )
    fams = [f for f in WF.FAMILY_ORDER if any(r["family"] == f for r in shown)]
    handles = [
        Line2D([], [], marker="o", ls="", color=WF.FAMILY_COLOR[f], label=WF.FAMILY_LABEL[f])
        for f in fams
    ] + [
        Line2D([], [], marker="o", ls="", color="0.35", label="drop vs PREFIX-only (filled)"),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            markerfacecolor="white",
            markeredgecolor="0.35",
            color="0.35",
            label="drop vs QUERY-only (hollow)",
        ),
    ]
    ax.legend(handles=handles, fontsize=7.4, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)

    (OUT / f"{STEM}.meta.json").write_text(
        json.dumps(
            {
                "what_is_plotted": (
                    "per predictor, the concordance of the per-feature R^2 DROP "
                    "(R2_context - R2_arm) on that predictor, for arm in {prefix, bare}"
                ),
                "arms_source": str(R2_DIR.relative_to(REPO)),
                "n_features": int(finite.sum()),
                "mean_drop": {
                    "prefix": float(np.nanmean(d_prefix)),
                    "bare": float(np.nanmean(d_bare)),
                },
                "top_n_drawn": TOP_N,
                "values": rows,
            },
            indent=1,
        )
    )
    print(f"[drop] wrote {OUT / (STEM + '.png')}")


if __name__ == "__main__":
    main()
