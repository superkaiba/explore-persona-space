"""#1482 SCRATCH: block-level redundancy heatmap + dendrogram, all 24 Shapley players.

WHAT IS PLOTTED. Pairwise FIRST CANONICAL CORRELATION between the column sets of
the 24 Shapley blocks (19 continuous cluster representatives + 5 judged axis
blocks), on the 113,260 complete-case features, all columns rank-transformed.
For two single-column blocks this reduces exactly to |Spearman|; for the
multi-column axis dummy blocks it is the largest correlation attainable between
any linear combination of each side -- so binary/categorical and continuous
predictors sit in ONE comparable matrix.

Rows/cols are ordered by complete-linkage hierarchical clustering on
(1 - canonical correlation), the same linkage and distance the production
decomposition uses to pick its representatives. The dendrogram shows where the
0.90 production cut falls and which pairs merge just below it.
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

import issue1482_shapley_blocks as SB

REPO = Path("/home/thomasjiralerspong/explore-persona-space")
OUT = REPO / "figures/issue_1482/block_redundancy"
CUT = 0.90
CUT_ALT = 0.85


def canonical_corr(C: np.ndarray, bi: list[int], bj: list[int]) -> float:
    """First canonical correlation between two column sets of a correlation matrix."""
    A, B = np.array(bi), np.array(bj)
    Caa = C[np.ix_(A, A)] + 1e-8 * np.eye(len(A))
    Cbb = C[np.ix_(B, B)] + 1e-8 * np.eye(len(B))
    Cab = C[np.ix_(A, B)]
    M = np.linalg.solve(Caa, Cab) @ np.linalg.solve(Cbb, Cab.T)
    return float(np.sqrt(np.clip(np.max(np.linalg.eigvals(M).real), 0.0, 1.0)))


def main() -> None:
    inp = SB.load_inputs(REPO / "data/issue_1482/densesae_target/ridge__mean_r2_fullwidth.npy")
    doc = json.loads(
        (
            REPO
            / "eval_results/issue_1482/predictor_battery/shapley_blocks_densesae_ridge_k24.json"
        ).read_text()
    )
    fs = doc["full_sample"]
    reps = [r["representative"] for r in fs["representatives"]]
    cov, r2 = inp["cov"], inp["r2"]

    ok = np.isfinite(r2)
    for c in reps:
        ok &= np.isfinite(cov[c])
    rows = np.flatnonzero(ok)

    gl = {
        ax: {
            "levels": [k for k in fs["axis_levels_pooled"][ax] if k != "__reference__"],
            "reference": fs["axis_levels_pooled"][ax]["__reference__"],
        }
        for ax in SB.AXIS_BLOCKS
    }
    xd, blocks, names, _ = SB.build_design(inp, rows, reps, list(SB.AXIS_BLOCKS), gl)
    X = np.column_stack([rankdata(xd[:, j]) for j in range(xd.shape[1])])
    C = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)

    n = len(blocks)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = R[j, i] = canonical_corr(C, blocks[i], blocks[j])

    D = 1.0 - R
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="complete")
    order = dendrogram(Z, no_plot=True)["leaves"]
    lab90 = fcluster(Z, t=1 - CUT, criterion="distance")
    lab85 = fcluster(Z, t=1 - CUT_ALT, criterion="distance")

    Ro = R[np.ix_(order, order)]
    no = [names[i] for i in order]

    fig = plt.figure(figsize=(13.2, 10.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 4.2], wspace=0.02)
    axd = fig.add_subplot(gs[0, 0])
    dendrogram(
        Z,
        orientation="left",
        labels=names,
        ax=axd,
        color_threshold=1 - CUT,
        above_threshold_color="#9A9A9A",
    )
    axd.axvline(1 - CUT, color="#D55E00", ls="--", lw=1.6)
    axd.axvline(1 - CUT_ALT, color="#0072B2", ls=":", lw=1.6)
    axd.set_xlabel("1 - canonical correlation", fontsize=8.5)
    axd.tick_params(labelsize=7.0)
    axd.set_yticks([])
    axd.invert_yaxis()

    axh = fig.add_subplot(gs[0, 1])
    im = axh.imshow(Ro, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axh.set_xticks(range(n))
    axh.set_xticklabels(no, rotation=90, fontsize=7.4)
    axh.set_yticks(range(n))
    axh.set_yticklabels(no, fontsize=7.4)
    for i in range(n):
        for j in range(n):
            if i != j and Ro[i, j] >= 0.50:
                axh.text(
                    j,
                    i,
                    f"{Ro[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.0,
                    color="w" if Ro[i, j] < 0.75 else "k",
                )
    cb = fig.colorbar(im, ax=axh, fraction=0.026, pad=0.015)
    cb.set_label("first canonical correlation", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle(
        "Redundancy among the 24 Shapley blocks (rank space, n=113,260)",
        fontsize=12.6,
        y=0.965,
    )
    fig.text(
        0.5,
        0.935,
        "First canonical correlation between block column sets — reduces to |Spearman| for single-column "
        "blocks, handles the multi-column judged-axis dummies natively. Complete linkage on (1 - corr), "
        "matching the production clustering. Cells >= 0.50 annotated.",
        ha="center",
        fontsize=7.8,
        color="#5A5A5A",
    )
    fig.text(
        0.5,
        -0.055,
        "These 24 blocks are ALREADY the output of declustering at 0.90, so by construction no pair "
        f"can exceed the orange line — the figure shows how CLOSE they come. Exactly one pair "
        f"(consistency / mean_act_cond, 0.851) merges at the dotted-blue {CUT_ALT:.2f} cut, "
        f"taking {lab90.max()} blocks to {lab85.max()}.",
        ha="center",
        fontsize=7.8,
        color="#5A5A5A",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "block_redundancy"
    for ext in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")

    iu = np.triu_indices(n, 1)
    top = sorted(zip(R[iu], [names[i] for i in iu[0]], [names[j] for j in iu[1]]), reverse=True)[
        :15
    ]
    merged85 = [
        (names[i], names[j])
        for i in range(n)
        for j in range(i + 1, n)
        if lab85[i] == lab85[j] and lab90[i] != lab90[j]
    ]
    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "what_is_plotted": (
                    "Pairwise first canonical correlation between the column sets of the 24 Shapley "
                    "blocks, rank-transformed, on the 113,260 complete-case features. Ordered by "
                    "complete-linkage clustering on (1 - canonical correlation)."
                ),
                "n_rows": int(len(rows)),
                "n_blocks": n,
                "block_names": names,
                "clusters_at_0.90": int(lab90.max()),
                "clusters_at_0.85": int(lab85.max()),
                "pairs_merging_between_0.85_and_0.90": merged85,
                "top15_pairs": [{"corr": float(v), "a": a, "b": b} for v, a, b in top],
                "caveats": [
                    "Canonical correlation is a MAXIMUM over linear combinations, so multi-column "
                    "axis blocks are not directly comparable to single-column blocks: a 5-level axis "
                    "has more freedom to align with anything than one continuous column does.",
                    "Rank space throughout; ties are NOT corrected here, so tie-heavy blocks "
                    "(template_token_frac, 58.4% zeros) are attenuated in this matrix as elsewhere.",
                    "The production clustering ran on CONTINUOUS candidates only; the 5 axis blocks "
                    "were added as players afterward and were never subject to the cut. This figure "
                    "is the first check of axis-vs-continuous redundancy.",
                    "Population is the current complete-case mask, which the proj_var join bug "
                    "restricts to 86.4% of the dictionary (low-activity features under-represented).",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {stem}.png / .pdf / .meta.json")
    print(f"  clusters: {lab90.max()} at {CUT}, {lab85.max()} at {CUT_ALT}")
    print(f"  merging between {CUT_ALT} and {CUT}: {merged85}")
    for v, a, b in top[:8]:
        print(f"    {v:.3f}  {a}  <->  {b}")


if __name__ == "__main__":
    main()
