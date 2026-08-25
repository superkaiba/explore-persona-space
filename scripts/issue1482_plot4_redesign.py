"""Paper-plan Plot 4 redesign (user directive 2026-08-24): two-panel figure.

Overwrites figures/paper/c3_sae_tier_gradient.{png,pdf,meta.json} (the name
plan.tex references). Replaces the old two-panel form (raw tier gradient +
answer-side activity quintile lines) per Thomas's spec: "first ... which
properties of the SAE feature predict best, controlling for all previous
features ... kind of like a list, and then on the right ... that Matryoshka
experiment, controlling for activity, and we don't need the answer side
activity quintile thing".

LEFT panel: RE-RENDER of the banked forward-stepwise concordance series
(figures/issue_1482/concordance/writeup_stepwise.meta.json; regular full-width
layer-19 SAE, n = 120,716 features; DV = dense-context->SAE-feature ridge
per-feature held-out R^2). One horizontal bar per selection round, value =
winner concordance c - 1/2 (0 = no association; positive = the property marks
BETTER-predicted features); each round's score conditions, via coarsened exact
matching, on every property selected above it. Nothing is recomputed.

RIGHT panel: the matryoshka tier gradient CONTROLLED FOR average activity,
computed inline from eval_results/issue_1482/matryoshka_tier/
perfeature_m_lmsys_default.npz: per-feature R^2 is centered on its activity
quintile's median (the SAME 5 strata the banked h1 within-stratum permutation
test conditions on), then summarized per tier as median with IQR whiskers.
Tier axis reads coarsest -> finest. The raw (unadjusted) gradient and the old
quintile panel are deliberately NOT drawn.

Usage:
    uv run python scripts/issue1482_plot4_redesign.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE numpy/matplotlib import

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    figsize_iclr_panels,
    paper_color,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

STEPWISE_META = PROJECT_ROOT / "figures/issue_1482/concordance/writeup_stepwise.meta.json"
PERFEATURE_NPZ = (
    PROJECT_ROOT / "eval_results/issue_1482/matryoshka_tier/perfeature_m_lmsys_default.npz"
)
TIER_TESTS = PROJECT_ROOT / "eval_results/issue_1482/matryoshka_tier/tier_tests.json"
OUT_EVAL = PROJECT_ROOT / "eval_results/issue_1482/plot4_redesign/plot4_redesign.json"
N_STRATA = 5  # activity quintiles: the banked h1 test's conditioning strata

# Plain-English canvas labels for the banked stepwise winner names, keyed on the
# EXACT banked strings so a re-run of the stepwise with renamed axes fails loud
# here instead of shipping a mislabeled bar. "Logit footprint: suppressing /
# promoting" are renamed per the standing reader-facing-rename directive
# (2026-08-21): the coinage never ships to a canvas.
WINNER_LABELS: dict[str, str] = {
    "Fires on BOTH context and answer side": "fires on both context and answer",
    "Mean activation (over all answers)": "mean activation (all answers)",
    "Speaker: identity / disposition": "speaker identity / disposition",
    "Logit footprint: suppressing": "suppresses specific output tokens",
    "Logit footprint: promoting": "promotes specific output tokens",
    "Content type: topic": "topic content",
    "Interpretable (autointerp)": "interpretable (auto-label)",
    "Judged role: output-promoting  [k=0.31]": "output-promoting role (noisy label)",
    "Side ratio (answer-side firing fraction)": "answer-side firing share",
    "Content type: task format": "task-format content",
    "Variance explained in answer space": "variance explained in answer space",
    "Content type: syntax": "syntax content",
    "SAE encoder norm (read strength)": "encoder norm (read strength)",
    "Mean activation (when active)": "mean activation (when active)",
}


def load_stepwise() -> list[dict]:
    """Banked stepwise rounds -> [{label, value, round}] in selection order.

    value = winner_c - 0.5 (signed concordance distance from the null)."""
    doc = json.loads(STEPWISE_META.read_text())
    rows = []
    for rnd in doc["rounds"]:
        name = rnd["winner"]
        if not name:  # the series' stop sentinel
            break
        assert name in WINNER_LABELS, f"unmapped banked winner name: {name!r}"
        rows.append(
            {
                "round": int(rnd["round"]),
                "banked_name": name,
                "label": WINNER_LABELS[name],
                "value": float(rnd["winner_c"]) - 0.5,
                "winner_c": float(rnd["winner_c"]),
                "pair_frac": float(rnd["pair_frac"]),
                "n": int(rnd["n"]),
            }
        )
    assert rows, "no banked stepwise rounds found"
    return rows


def _strata_of(activity: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile strata over activity (the matryoshka driver's recipe verbatim)."""
    edges = np.quantile(activity, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, activity, side="right")


def tier_gradient_activity_controlled() -> dict:
    """Per-tier median + IQR of R^2 centered within activity quintiles.

    Returns raw AND adjusted per-tier summaries plus a pooled adjusted Spearman,
    so the committed JSON carries the before/after-control comparison."""
    z = np.load(PERFEATURE_NPZ)
    r2_all = np.asarray(z["r2"], np.float64)
    ok = np.isfinite(r2_all)
    r2 = r2_all[ok]
    tier = np.asarray(z["tier"], np.int64)[ok]
    act = np.asarray(z["activity"], np.float64)[ok]
    strata = _strata_of(act, N_STRATA)
    centered = r2.copy()
    for s in range(N_STRATA):
        m = strata == s
        assert m.any(), f"empty activity stratum {s}"
        centered[m] -= np.median(r2[m])
    from scipy.stats import spearmanr

    out: dict = {
        "n_features_finite": int(ok.sum()),
        "n_strata": N_STRATA,
        "strata": "activity quintiles (quantile edges over the finite-R^2 panel)",
        "centering": "per-feature R^2 minus the median R^2 of its activity quintile",
        "spearman_tier_r2_raw": float(spearmanr(tier, r2).statistic),
        "spearman_tier_r2_activity_centered": float(spearmanr(tier, centered).statistic),
        "per_tier": {},
    }
    for t in (0, 1, 2):
        v_raw = r2[tier == t]
        v_adj = centered[tier == t]
        out["per_tier"][str(t)] = {
            "n": int((tier == t).sum()),
            "median_raw": float(np.median(v_raw)),
            "median_adjusted": float(np.median(v_adj)),
            "q25_adjusted": float(np.percentile(v_adj, 25)),
            "q75_adjusted": float(np.percentile(v_adj, 75)),
        }
    return out


def render(rows: list[dict], tier_doc: dict) -> None:
    """Two-panel figure: ranked property list (left), controlled tier gradient (right)."""
    set_paper_style("iclr")
    fig = plt.figure(figsize=figsize_iclr_panels(2, height_in=2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0), wspace=0.52)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    ys = np.arange(len(rows))[::-1]  # round 0 at the top
    vals = [r["value"] for r in rows]
    ax_l.barh(ys, vals, height=0.62, color=paper_color("instruct"))
    ax_l.axvline(0.0, color=paper_color("null"), lw=0.8)
    ax_l.set_yticks(ys, [r["label"] for r in rows], fontsize=6)
    ax_l.set_ylim(-0.6, len(rows) - 0.4)
    ax_l.set_xlabel("concordance with per-feature $R^2$ $-$ 1/2", fontsize=7)
    ax_l.set_title("feature properties, ranked\n(each bar controls for those above)", fontsize=7)
    ax_l.tick_params(axis="x", labelsize=6)

    cmap = matplotlib.colormaps["viridis"]
    tier_col = {t: cmap(v) for t, v in zip((0, 1, 2), (0.15, 0.5, 0.8), strict=True)}
    for t in (0, 1, 2):
        d = tier_doc["per_tier"][str(t)]
        med = d["median_adjusted"]
        lo = max(0.0, med - d["q25_adjusted"])
        hi = max(0.0, d["q75_adjusted"] - med)
        ax_r.errorbar([t], [med], yerr=[[lo], [hi]], fmt="o", ms=4, capsize=2, color=tier_col[t])
    ax_r.axhline(0.0, color=paper_color("null"), ls=":", lw=1)
    ax_r.set_xticks([0, 1, 2], ["coarsest", "mid", "finest"])
    ax_r.set_xlim(-0.6, 2.6)
    ax_r.set_xlabel("matryoshka tier (coarsest to finest)", fontsize=7)
    ax_r.set_ylabel("per-feature $R^2$, centered within\nactivity quintiles", fontsize=7)
    ax_r.set_title("tier gradient, controlling for activity", fontsize=7)
    ax_r.tick_params(labelsize=6)

    savefig_paper(fig, "c3_sae_tier_gradient", dir="figures/paper/")
    plt.close(fig)
    print("[plot4] wrote figures/paper/c3_sae_tier_gradient.{png,pdf,meta.json}", flush=True)


def main() -> int:
    """Load banked inputs, compute the controlled gradient, write JSON + figure."""
    rows = load_stepwise()
    tier_doc = tier_gradient_activity_controlled()
    banked_tests = json.loads(TIER_TESTS.read_text())
    OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "left_panel": {
            "source": str(STEPWISE_META.relative_to(PROJECT_ROOT)),
            "statistic": "winner concordance c - 1/2 per forward-selection round; each "
            "round conditions (coarsened exact matching) on all previously selected "
            "properties",
            "dv": "dense-context->SAE-feature ridge per-feature held-out R^2 "
            "(regular full-width layer-19 SAE; banked in the stepwise sidecar)",
            "rows": rows,
        },
        "right_panel": {
            "source": str(PERFEATURE_NPZ.relative_to(PROJECT_ROOT)),
            "banked_reference": {
                "h1_partial_spearman_tier_r2_given_logact": banked_tests["h1_tier_within_stratum"][
                    "partial_spearman_tier_r2_given_logact"
                ],
                "h1_perm_band_2p5_97p5": banked_tests["h1_tier_within_stratum"][
                    "perm_band_2p5_97p5"
                ],
                "h1_observed_pooled_spearman": banked_tests["h1_tier_within_stratum"][
                    "observed_pooled_spearman"
                ],
            },
            **tier_doc,
        },
        "dropped": "answer-side activity quintile panel (user directive 2026-08-24)",
        "metadata": as_metadata_dict(git_provenance(), phase="plot4-redesign"),
    }
    OUT_EVAL.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[plot4] wrote {OUT_EVAL.relative_to(PROJECT_ROOT)}", flush=True)
    render(rows, tier_doc)
    for r in rows:
        print(f"[plot4] round {r['round']:2d}  c-1/2 = {r['value']:+.3f}  {r['label']}")
    for t in (0, 1, 2):
        d = tier_doc["per_tier"][str(t)]
        print(
            f"[plot4] tier {t}: median_raw = {d['median_raw']:+.4f}  "
            f"median_adjusted = {d['median_adjusted']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
