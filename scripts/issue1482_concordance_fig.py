"""#1482 SCRATCH: concordance (Harrell c-index) of every predictor with per-feature R^2.

WHAT IS PLOTTED. For each predictor, Somers' D of R^2 on that predictor, reported on
the AUROC / c-index scale:

    D_{y|x} = (C - Dis) / (n_pairs - T_x)        c = (D + 1) / 2

the probability that, in a random pair of features the PREDICTOR separates, the one
with the higher predictor value is the better-predicted one. 0.5 = chance.

For a binary predictor this reduces EXACTLY to the ordinary AUROC (verified
bit-identical against sklearn.roc_auc_score with the indicator as label and R^2 as
score); for a continuous predictor it is the rank-concordance generalisation. R^2 is
never thresholded, and dividing by predictor-untied pairs removes the prevalence
ceiling that pins a rare indicator near chance under a median-split AUROC.

FOUR VALUES PER PREDICTOR -- the same C/Dis/T_x counts accumulated WITHIN strata and
pooled, so pairs only ever form between comparable features. Nonparametric
conditioning: no functional form is assumed for the firing-rate -> R^2 relationship.

  pooled     all pairs
  m_token    10 deciles of firing_freq_per_token  (feature density, per token)
  m_answer   10 deciles of activity               (firing frequency, per answer)
  m_both     nested: 10 per-token deciles x 3 within-decile activity terciles

m_both is NESTED rather than a 10x10 crossing because rho(activity, per-token) = 0.96
-- a crossed grid would leave the off-diagonal cells empty, while nesting holds all 30
cells at ~4,000 features each.

The FIGURE plots pooled vs m_both (the strictest control). The gap between them is the
part of the association that runs through firing rate in either sense; all four values
are recorded in the sidecar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/thomasjiralerspong/explore-persona-space/scripts")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/scipy/matplotlib import

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import kendalltau, rankdata

import issue1482_shapley_blocks as SB

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
OUT = REPO / "figures/issue_1482/concordance"
COV_V2 = "eval_results/issue_1482/predictor_battery/fullwidth_covariates_v2.npz"
CTRL = "firing_freq_per_token"
CTRL2 = "activity"
NBINS = 10
MIN_POS = 60

C_CONT, C_BIN, C_CTRL = "#0072B2", "#D55E00", "#4D4D4D"  # Okabe-Ito


def cd_untied(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(C - Dis, n_pairs_untied_in_x) for one sample. Exact for binary x."""
    m = len(y)
    if m < 2 * MIN_POS:
        return 0.0, 0.0
    vals, counts = np.unique(x, return_counts=True)
    if len(vals) < 2:
        return 0.0, 0.0
    n0 = m * (m - 1) / 2
    tx = float(np.sum(counts * (counts - 1) / 2))
    den = n0 - tx
    if den <= 0:
        return 0.0, 0.0
    if len(vals) == 2:  # closed-form Mann-Whitney
        pos = x == vals[1]
        k = int(pos.sum())
        r = rankdata(y)
        auc = (r[pos].sum() - k * (k + 1) / 2) / (k * (m - k))
        return (2 * auc - 1) * den, den
    tb = kendalltau(x, y, variant="b").statistic
    _, cy = np.unique(y, return_counts=True)
    ty = float(np.sum(cy * (cy - 1) / 2))
    return tb * np.sqrt((n0 - tx) * (n0 - ty)), den


def concordance(x: np.ndarray, y: np.ndarray, strata: list[np.ndarray]) -> float:
    num = den = 0.0
    for idx in strata:
        a, d = cd_untied(x[idx], y[idx])
        num += a
        den += d
    return 0.5 * (num / den + 1) if den > 0 else np.nan


def main() -> None:
    SB.COV_NPZ = COV_V2  # proj_var-corrected universe
    inp = SB.load_inputs(REPO / "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy")
    doc = json.loads(
        (
            REPO
            / "eval_results/issue_1482/predictor_battery/shapley_blocks_densesae_ridge_k24.json"
        ).read_text()
    )
    reps = [r["representative"] for r in doc["full_sample"]["representatives"]]
    cov, lab, r2 = inp["cov"], inp["labels"], inp["r2"]

    ok = np.isfinite(r2) & np.isfinite(cov[CTRL]) & np.isfinite(cov[CTRL2])
    for c in reps:
        ok &= np.isfinite(cov[c])
    y = r2[ok]
    F = cov[CTRL][ok]
    FA = cov[CTRL2][ok]
    n = len(y)

    def deciles(v: np.ndarray) -> np.ndarray:
        e = np.percentile(v, np.linspace(0, 100, NBINS + 1))
        return np.clip(np.digitize(v, e[1:-1]), 0, NBINS - 1)

    ALL = [np.arange(n)]
    S_TOKEN = [np.flatnonzero(deciles(F) == j) for j in range(NBINS)]
    S_ANSWER = [np.flatnonzero(deciles(FA) == j) for j in range(NBINS)]
    S_BOTH = []
    for idx in S_TOKEN:  # nest activity terciles inside each
        q = np.percentile(FA[idx], [100 / 3, 200 / 3])
        inner = np.clip(np.digitize(FA[idx], q), 0, 2)
        for t in range(3):
            S_BOTH.append(idx[inner == t])

    # ---- assemble predictors --------------------------------------------------
    items: list[tuple[str, str, int, np.ndarray]] = []
    for c in reps:
        if c == CTRL:
            continue
        v = cov[c][ok]
        kind = "binary" if len(np.unique(v)) == 2 else "continuous"
        npos = int((v == np.unique(v)[1]).sum()) if kind == "binary" else n
        items.append((c, kind, npos, v))
    for axis in sorted(lab):
        a = np.asarray(lab[axis])[ok]
        for lv in sorted(set(a.tolist())):
            if lv == "unlabeled":
                continue
            m = (a == lv).astype(float)
            k = int(m.sum())
            if MIN_POS < k < n - MIN_POS:
                items.append((f"{axis}:{lv}", "binary", k, m))
    for lv in sorted(set(cov["side_class"][ok].tolist())):
        m = (cov["side_class"][ok] == lv).astype(float)
        k = int(m.sum())
        if MIN_POS < k < n - MIN_POS:
            items.append((f"side_class={int(lv)}", "binary", k, m))

    rows = [
        {
            "predictor": nm,
            "kind": kd,
            "n_pos": k,
            "pooled": concordance(v, y, ALL),
            "m_token": concordance(v, y, S_TOKEN),
            "m_answer": concordance(v, y, S_ANSWER),
            "matched": concordance(v, y, S_BOTH),
        }
        for nm, kd, k, v in items
    ]
    ctrl_pooled = concordance(F, y, ALL)
    ctrl2_pooled = concordance(FA, y, ALL)
    rows.sort(key=lambda r: r["matched"])  # ascending -> best at top

    # ---- figure ---------------------------------------------------------------
    labels = [r["predictor"] for r in rows]
    pooled = np.array([r["pooled"] for r in rows])
    matched = np.array([r["matched"] for r in rows])
    kinds = [r["kind"] for r in rows]
    yy = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(9.6, 0.265 * len(rows) + 2.9))
    ax.axvline(0.5, color="#999999", lw=1.3, zorder=1)
    ax.axvline(ctrl_pooled, color=C_CTRL, ls="--", lw=1.3, zorder=1)

    for i, (p, m, kd) in enumerate(zip(pooled, matched, kinds)):
        col = C_BIN if kd == "binary" else C_CONT
        ax.plot([p, m], [i, i], color=col, lw=1.5, alpha=0.42, zorder=2, solid_capstyle="round")
        ax.scatter(p, i, s=27, facecolors="white", edgecolors=col, lw=1.35, zorder=3)
        ax.scatter(m, i, s=40, color=col, zorder=4)

    ax.set_yticks(yy)
    ax.set_yticklabels(
        [
            f"{r['predictor']}" + (f"  (n={r['n_pos']:,})" if r["kind"] == "binary" else "")
            for r in rows
        ],
        fontsize=7.5,
    )
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel(
        "concordance (Harrell c-index) with per-feature $R^2$  —  0.5 = chance", fontsize=9.4
    )
    ax.tick_params(axis="x", labelsize=8.2)
    ax.grid(axis="x", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)

    ax.set_title(
        "How well does each predictor rank features by how well the map predicts them?\n"
        "open = pooled · filled = matched on firing rate (per-token decile × per-answer tercile) · "
        f"n = {n:,} features",
        fontsize=11.3,
    )
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                mfc="white",
                mec=C_CONT,
                mew=1.35,
                label="continuous — pooled",
            ),
            Line2D([], [], marker="o", ls="", color=C_CONT, label="continuous — matched"),
            Line2D(
                [], [], marker="o", ls="", mfc="white", mec=C_BIN, mew=1.35, label="binary — pooled"
            ),
            Line2D([], [], marker="o", ls="", color=C_BIN, label="binary — matched"),
            Line2D(
                [],
                [],
                ls="--",
                color=C_CTRL,
                label=f"per-token firing rate itself ({ctrl_pooled:.3f})",
            ),
        ],
        fontsize=7.6,
        loc="lower right",
        framealpha=0.94,
    )
    fig.text(
        0.5,
        -0.028 - 0.0006 * len(rows),
        "c = P(the feature with the higher predictor value is the better-predicted one), over pairs the "
        "predictor separates.\nFor a binary predictor this is exactly the ordinary AUROC; $R^2$ is never "
        "thresholded. MATCHED forms pairs only within cells of\nper-token firing decile × within-decile "
        "per-answer activity tercile (30 cells, ~4,000 features each). Every value is MARGINAL: "
        "firing rate is controlled, the other predictors are not (pairwise correlations reach 0.85). "
        "No confidence intervals.",
        ha="center",
        fontsize=7.4,
        color="#5A5A5A",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "concordance_pooled_vs_matched"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")

    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "what_is_plotted": (
                    "Somers' D of per-feature R^2 on each predictor, on the AUROC / Harrell c-index "
                    "scale: the probability that, in a random pair of features the predictor separates, "
                    "the one with the higher predictor value is the better-predicted one. Two values per "
                    "predictor: pooled over all pairs, and matched (pairs formed only within per-token "
                    "firing-rate deciles)."
                ),
                "statistic": "D_{y|x} = (C - Dis) / (n_pairs - T_x);  c = (D + 1) / 2",
                "equivalence_check": (
                    "For binary predictors this is bit-identical to "
                    "sklearn.metrics.roc_auc_score(indicator, r2) — verified to 1e-16 on "
                    "speaker_property:identity_disposition, speaker_property:language, "
                    "abstraction:abstract_contextual."
                ),
                "n_rows": int(n),
                "covariates": COV_V2,
                "r2_target": "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy",
                "control": CTRL,
                "control_2": CTRL2,
                "matched_scheme": (
                    "nested: 10 deciles of firing_freq_per_token x 3 within-decile terciles of "
                    "activity (30 cells). Nested rather than crossed because "
                    "rho(activity, firing_freq_per_token) = 0.96 empties a crossed grid's "
                    "off-diagonal."
                ),
                "n_bins": NBINS,
                "bin_n": [int(len(i)) for i in S_TOKEN],
                "bin_median_firing": [float(np.median(F[i])) for i in S_TOKEN],
                "both_cell_n": [int(len(i)) for i in S_BOTH],
                "control_pooled_concordance": float(ctrl_pooled),
                "control_2_pooled_concordance": float(ctrl2_pooled),
                "values": rows,
                "caveats": [
                    "MARGINAL, not incremental: firing rate is matched, the other predictors are NOT. "
                    "Pairwise block correlations reach 0.85, so a strong value may be a correlated "
                    "neighbour showing through.",
                    "No confidence intervals. Differences under ~0.02 should not be read; rare "
                    "indicators (n_pos < ~1,500) are the noisiest rows.",
                    "Both firing-rate senses are matched, but the rest of the activity family is not: "
                    "mean_act_cond correlates only 0.40 with activity and is NOT well matched by this "
                    "design.",
                    "dense_latent_flag concentrates in the top firing deciles, so its matched value "
                    "rests on comparatively few within-bin pairs.",
                    "functional_role and gurnee_promoting_class were EXCLUDED from the 24-block Shapley "
                    "decomposition (kappa = 0.310 for functional_role; gurnee is a q0.90 slice off "
                    "footprint_kurt/skew). functional_role levels appear here for completeness and "
                    "should not carry a headline.",
                    "Population is the v2 (proj_var-corrected) covariate universe, complete-case over "
                    "the 19 continuous cluster representatives.",
                ],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"wrote {stem}.png / .pdf / .meta.json   ({len(rows)} predictors, n={n:,})")
    print(f"  {CTRL} itself (pooled): {ctrl_pooled:.3f}\n")
    print(f"  {'predictor':38s} {'pooled':>7} {'matched':>8} {'shift':>7}")
    for r in sorted(rows, key=lambda r: -abs(r["matched"] - 0.5))[:12]:
        print(
            f"  {r['predictor']:38s} {r['pooled']:7.3f} {r['matched']:8.3f} "
            f"{r['matched'] - r['pooled']:+7.3f}"
        )


if __name__ == "__main__":
    main()
