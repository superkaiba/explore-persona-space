"""Conf1 confirmation forest plot for issue #2094 follow-up round `fu1_regen_confirm`.

The parent grid's 15 clean null-separated behavior families were re-measured
independently at temperature 1.0 with 5 draws per pair (CONF-1), steered vs the
parent's seeded shuffled-donor null. This figure shows, per family, the steered
and null means with pair-clustered bootstrap 95 percent intervals (B = 10,000)
plus the per-pair mean-F values behind each interval (the low-level companion),
with confirmed families (fully disjoint intervals, steered above) drawn filled
and unconfirmed families drawn open.

Family-level means/CIs/verdicts come verbatim from the committed
eval_results/issue_2094/f_metrics/fu1/fu1_conf1_confirmation.json (never
recomputed); the per-pair points are recomputed through the fu1 analysis
module's own reduction (`issue2094_fu1_analysis.reduce_conf1`) and tied to the
committed means by a nanmean consistency assert.

Writes figures/issue_2094/fu1_conf1_forest.{png,pdf,meta.json}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue2094_fu1 as FU1  # noqa: E402
import issue2094_fu1_analysis as FA  # noqa: E402
import issue2094_wellsep_bootstrap as W  # noqa: E402

FMETRICS = Path("eval_results/issue_2094/f_metrics")
CONF1_PATH = FMETRICS / "fu1/fu1_conf1_confirmation.json"

STEERED_COLOR = paper_palette_role("primary")
NULL_COLOR = paper_palette_role("baseline")

_LV_LABEL = {"joint_mid": "mid-stack joint", "joint_all": "all 28 layers"}
_SETTING_LABEL = {"matched_query": "matched query", "cross": "cross"}


def family_label(family: str) -> str:
    """Plain-English y-tick label for a conf1 family key
    ``setting|slot|layer_variant|dose|vec_type|metric`` (all context-end)."""
    setting, slot, lv, dose, vt, _metric = family.split("|")
    assert slot == "ce", family
    lv_txt = _LV_LABEL.get(lv, f"layer {lv.removeprefix('L')}")
    dose_txt = "full-state patch" if dose == "replace" else f"dose {dose.removeprefix('a')}x"
    label = f"{_SETTING_LABEL[setting]} - {lv_txt}, {dose_txt}"
    if vt == "B":
        label += " (prefix centroid)"
    return label


def per_pair_values() -> dict[str, dict[str, np.ndarray]]:
    """Per (family, arm) per-pair mean-F values via the fu1 analysis module's
    own reduction (identical conventions to the committed artifact)."""
    scores_dir = FA.DEFAULT_SCORES_DIR
    FA.check_wave_metas(scores_dir)
    rows_iter = (
        row for f in sorted(scores_dir.glob("*.scores.jsonl")) for row in FA.A._iter_jsonl(f)
    )
    sc = FA.route_fu1_scores(rows_iter)
    anchors = FA.load_anchor_stats(FMETRICS / "anchors.jsonl")
    fragility = json.loads(Path(FU1.FRAGILITY_REL).read_text(encoding="utf-8"))
    parent_wellsep = json.loads((FMETRICS / "bootstrap_cis_wellsep.json").read_text("utf-8"))
    breached = set(FU1.derive_breached_cells(fragility))
    families = FU1.derive_conf1_families(parent_wellsep, breached)
    ws, _ws_any = W.load_wellsep(FMETRICS / "anchors.jsonl", W.MIN_SEPARATION)
    reduced = FA.reduce_conf1(sc, families, anchors, ws)
    return {
        fam: {arm: rec["arms"][arm]["values"] for arm in ("steered", "null")}
        for fam, rec in reduced.items()
    }


def main() -> int:
    conf1 = json.loads(CONF1_PATH.read_text(encoding="utf-8"))
    fams = sorted(conf1["families"], key=lambda f: -f["steered"]["observed_mean"])
    pair_vals = per_pair_values()

    # Tie the recomputed per-pair values to the committed family means.
    for f in fams:
        for arm in ("steered", "null"):
            got = float(np.nanmean(pair_vals[f["family"]][arm]))
            want = f[arm]["observed_mean"]
            assert abs(got - want) < 1e-9, (f["family"], arm, got, want)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.4, 7.2), layout="constrained")
    rng = np.random.default_rng(42)
    ys = np.arange(len(fams))[::-1]  # largest steered mean at the top

    for f, y in zip(fams, ys):
        for arm, color, dy in (("steered", STEERED_COLOR, +0.18), ("null", NULL_COLOR, -0.18)):
            rec = f[arm]
            v, lo, hi = rec["observed_mean"], rec["ci_lo"], rec["ci_hi"]
            vals = pair_vals[f["family"]][arm]
            vals = vals[np.isfinite(vals)]
            jitter = rng.uniform(-0.06, 0.06, size=vals.size)
            ax.scatter(
                vals,
                np.full(vals.size, y + dy) + jitter,
                s=9,
                color=color,
                alpha=0.35,
                linewidths=0,
                zorder=2,
            )
            filled = arm == "null" or f["confirmed"]
            ax.errorbar(
                [v],
                [y + dy],
                xerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.6,
                capsize=2.5,
                markersize=6 if arm == "steered" else 4.5,
                mfc=color if filled else "white",
                mec=color,
                mew=1.4,
                zorder=3,
            )

    ax.axvline(0.0, color="grey", linewidth=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([family_label(f["family"]) for f in fams], fontsize=8.5)
    ax.set_xlabel("behavior F at temperature 1.0 (fraction of a full context swap, prefix rubric)")
    ax.set_title(
        "independent re-sampling of the 15 surviving families: steered vs shuffled-donor null",
        loc="left",
        fontsize=10,
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=STEERED_COLOR,
            markersize=6,
            label="steered mean (confirmed: 95 percent intervals disjoint)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            mfc="white",
            mec=STEERED_COLOR,
            mew=1.4,
            color=STEERED_COLOR,
            markersize=6,
            label="steered mean (not confirmed)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=NULL_COLOR,
            markersize=4.5,
            label="shuffled-donor null mean",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="dimgray",
            markersize=3,
            alpha=0.5,
            label="per-pair means (5 draws each, coherent kept draws)",
        ),
    ]
    fig.legend(handles=handles, fontsize=7.5, loc="outside lower center", ncols=2)
    savefig_paper(fig, "issue_2094/fu1_conf1_forest", dir="figures/")
    plt.close(fig)
    print("[fu1-fig] wrote figures/issue_2094/fu1_conf1_forest.{png,pdf,meta.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
