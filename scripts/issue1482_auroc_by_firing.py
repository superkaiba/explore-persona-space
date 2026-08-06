"""#1482 SCRATCH: AUROC of every predictor WITHIN firing-rate bins.

WHAT IS PLOTTED. For each predictor (19 continuous cluster representatives + the
judged-axis levels as binary indicators) and each of 10 equal-count bins of
per-token firing frequency: the AUROC with which that predictor discriminates
above-median-R^2 features from below-median-R^2 features, where the median is
taken WITHIN the bin. 0.5 = no discrimination.

Binarising R^2 at the within-bin median puts continuous and binary predictors on
ONE comparable scale: for a binary indicator this is the ordinary label AUROC,
for a continuous predictor it is the concordance (a monotone transform of the
within-bin Spearman).

Binning on firing rate is NONPARAMETRIC matching: unlike partialling, it assumes
nothing about the functional form of the firing-rate -> R^2 relationship.
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
from scipy.stats import rankdata

import issue1482_shapley_blocks as SB

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
OUT = REPO / "figures/issue_1482/auroc_by_firing"
NBINS = 10
MIN_POS = 60


def auroc(score: np.ndarray, pos: np.ndarray) -> float:
    npos = int(pos.sum())
    nneg = len(pos) - npos
    if npos < MIN_POS or nneg < MIN_POS:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main() -> None:
    inp = SB.load_inputs(REPO / "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy")
    doc = json.loads(
        (
            REPO
            / "eval_results/issue_1482/predictor_battery/shapley_blocks_densesae_ridge_k24.json"
        ).read_text()
    )
    reps = [r["representative"] for r in doc["full_sample"]["representatives"]]
    cov, lab, r2 = inp["cov"], inp["labels"], inp["r2"]

    ok = np.isfinite(r2) & np.isfinite(cov["firing_freq_per_token"])
    for c in reps:
        ok &= np.isfinite(cov[c])

    F = cov["firing_freq_per_token"][ok]
    y = r2[ok]
    edges = np.percentile(F, np.linspace(0, 100, NBINS + 1))
    b = np.clip(np.digitize(F, edges[1:-1]), 0, NBINS - 1)

    preds: list[tuple[str, np.ndarray]] = [(c, cov[c][ok]) for c in reps]
    for axis in sorted(lab):
        a = np.asarray(lab[axis])[ok]
        for lv in sorted(set(a.tolist())):
            if str(lv) in ("unlabeled",):
                continue
            m = (a == lv).astype(float)
            if MIN_POS < m.sum() < len(m) - MIN_POS:
                preds.append((f"{axis[:11]}:{lv}"[:30], m))

    M = np.full((len(preds), NBINS), np.nan)
    for j in range(NBINS):
        sel = b == j
        med = np.median(y[sel])
        hi = y[sel] > med
        for i, (_, v) in enumerate(preds):
            M[i, j] = auroc(v[sel], hi)

    strength = np.nanmean(np.abs(M - 0.5), axis=1)
    order = np.argsort(-strength)
    M, names = M[order], [preds[i][0] for i in order]

    keep = ~np.isnan(strength[order]) & (strength[order] > 0.005)
    M, names = M[keep], [n for n, k in zip(names, keep) if k]

    fig, ax = plt.subplots(figsize=(11.4, 0.30 * len(names) + 3.0))
    vmax = float(np.nanmax(np.abs(M - 0.5)))
    im = ax.imshow(M, cmap="PuOr_r", vmin=0.5 - vmax, vmax=0.5 + vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]) and abs(M[i, j] - 0.5) > 0.06:
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.1,
                    color="w" if abs(M[i, j] - 0.5) > vmax * 0.62 else "k",
                )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7.4)
    ax.set_xticks(range(NBINS))
    ax.set_xticklabels(
        [f"{j + 1}\nn={int((b == j).sum()):,}\n{np.median(F[b == j]):.1e}" for j in range(NBINS)],
        fontsize=6.6,
    )
    ax.set_xlabel(
        "per-token firing-rate bin (equal count; n and median rate annotated)", fontsize=9
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cb.set_label("AUROC: discriminates above- vs below-median $R^2$ WITHIN the bin", fontsize=8.2)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_title(
        "Which predictors work at matched firing rate?\n"
        "AUROC within per-token firing-rate bins — 0.5 = no discrimination; "
        "sorted by mean |AUROC - 0.5|",
        fontsize=11.6,
    )
    fig.text(
        0.5,
        -0.035,
        "Binning on firing rate is NONPARAMETRIC matching (assumes no functional form), unlike partialling. "
        "R^2 is binarised at the WITHIN-BIN median, so continuous and binary predictors share one scale. "
        "Each cell is a MARGINAL association: other predictors are NOT controlled, so collinear predictors "
        "can show each other through.",
        ha="center",
        fontsize=7.6,
        color="#5A5A5A",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "auroc_by_firing"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")

    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "what_is_plotted": (
                    "AUROC of each predictor discriminating above- vs below-median per-feature R^2 "
                    "within each of 10 equal-count per-token firing-rate bins."
                ),
                "n_rows": int(ok.sum()),
                "n_bins": NBINS,
                "bin_n": [int((b == j).sum()) for j in range(NBINS)],
                "bin_median_firing": [float(np.median(F[b == j])) for j in range(NBINS)],
                "predictors": names,
                "auroc": {
                    n: [None if np.isnan(v) else round(float(v), 4) for v in row]
                    for n, row in zip(names, M)
                },
                "caveats": [
                    "MARGINAL, not incremental: each cell controls firing rate (by matching) but NOT the "
                    "other predictors. Given pairwise correlations up to 0.85, a strong cell may be another "
                    "predictor showing through.",
                    "Binarising R^2 at the within-bin median discards magnitude and costs power; it is done "
                    "to put binary and continuous predictors on one scale.",
                    "Matched on per-token firing only. rho(per-answer, per-token) = 0.96 so the rest of the "
                    "activity family is largely but not exactly matched; mean_act_cond is only 0.40 with "
                    "activity and is NOT well matched by this design.",
                    "No confidence intervals. Cells near 0.5 are not distinguishable from chance here.",
                    "v2 (proj_var-corrected) covariate universe.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png  ({len(names)} predictors x {NBINS} bins, n={int(ok.sum()):,})")
    for n, row in list(zip(names, M))[:8]:
        print(f"  {n:30s} " + " ".join(f"{v:.2f}" if np.isfinite(v) else " -- " for v in row))


if __name__ == "__main__":
    main()
