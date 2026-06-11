"""Issue #537 analyzer figures, part 2 (round 1).

Generates the per-finding figures not covered by i537_analyzer_figures.py:
  6. behavior_dependence  -- 5x5 cross-behavior Spearman heatmap of off-diag G
  7. asymmetry_g1         -- antisym fraction bars + raw G1 scatters (both regressors)
  8. g2_parallelism       -- per-layer parallelism vs floor/ceiling + scaling rho strip
  9. em_contrastive_nc    -- contrastive vs non-contrastive EM off-diag leakage

Saves into the worktree's figures/issue_537/ (committed on the issue-537 branch).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

EVAL = Path("eval_results/issue_537")
FIGDIR = "figures/"

BEH_LABELS = {
    "marker": "Marker tic ※",
    "fact": "Taught fact",
    "refusal": "Blanket refusal",
    "sycophancy": "Sycophancy",
    "em": "Harmful advice",
}
BEH_ORDER = ["marker", "fact", "refusal", "sycophancy", "em"]


def fig_behavior_dependence():
    """5x5 Spearman heatmap between the off-diagonal G matrices of the 5 behaviors."""
    with open(EVAL / "analysis/registered_reads.json") as f:
        reads = json.load(f)
    pw = reads["h_behavior_dependence"]["pairwise"]
    n = len(BEH_ORDER)
    M = np.eye(n)
    for k, v in pw.items():
        a, b = k.split("~")
        i, j = BEH_ORDER.index(a), BEH_ORDER.index(b)
        M[i, j] = M[j, i] = v["rho"]
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.92, top=0.82, bottom=0.16)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    # round 2: visually flag the noise-limited refusal row/column (failed the
    # h_structure floor; its correlations are texture, not data)
    r_idx = BEH_ORDER.index("refusal")
    for k in range(n):
        for i, j in ((r_idx, k), (k, r_idx)):
            ax.add_patch(
                mpl.patches.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    facecolor="white",
                    alpha=0.55,
                    edgecolor="0.55",
                    linewidth=0.0,
                    hatch="///",
                    zorder=2,
                )
            )
    for i in range(n):
        for j in range(n):
            faded = r_idx in (i, j)
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="0.55" if faded else ("white" if abs(M[i, j]) > 0.6 else "0.2"),
                zorder=3,
            )
    labels = [BEH_LABELS[b] + ("\n(noise-limited)" if b == "refusal" else "") for b in BEH_ORDER]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        "Different behaviors generalize along different context structure\n"
        "Rank correlation between off-diagonal G matrices (464 cells per behavior);\n"
        "hatched = the noise-limited refusal row (implants failed; excluded from\n"
        "the headline read)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.grid(False)
    savefig_paper(fig, "issue_537/behavior_dependence", dir=FIGDIR)
    plt.close(fig)


SHARED = [
    "sp_swe",
    "sp_doctor",
    "sp_ph1",
    "sp_ph2",
    "wc_short_code",
    "wc_short_advice",
    "wc_long_write",
    "icl_k2",
    "icl_k8",
    "reph_imp",
    "reph_polite",
    "reph_casual",
    "fmt_json",
    "fmt_code",
    "default",
    "binst_marker",
]
EXCLUDE_PRIMARY = {"fmt_code", "binst_marker"}


def _g1_points():
    """Recompute the raw G1 scatter points (same logic as i537_g1_regression.py)."""
    norms = {}
    for cid in SHARED:
        z = np.load(EVAL / "clouds" / f"{cid}__mean_response.npz")
        h = z["hidden"][:, 22, :].astype(np.float64)
        h = h[np.isfinite(h).all(axis=1)]
        norms[cid] = float(np.linalg.norm(h.mean(axis=0)))
    cells = {}
    for ci in SHARED:
        for cj in SHARED:
            p = EVAL / f"G_cells/marker/{ci}__{cj}__seed42.json"
            if p.exists():
                with open(p) as f:
                    cell = json.load(f)
                cells[(ci, cj)] = np.mean(
                    [q["trained"]["logp"] - q["base"]["logp"] for q in cell["per_question"]]
                )
    s = {ci: cells[(ci, ci)] for ci in SHARED}
    with open(EVAL / "prereg/quarantine_manifest.json") as f:
        qm = json.load(f)
    quar = {tuple(c) for c in qm["quarantined_cells"]["marker"]}
    ctx = [c for c in SHARED if c not in EXCLUDE_PRIMARY]
    y, x1, x2 = [], [], []
    for a in range(len(ctx)):
        for b in range(a + 1, len(ctx)):
            ci, cj = ctx[a], ctx[b]
            if (ci, cj) in quar or (cj, ci) in quar:
                continue
            if (ci, cj) in cells and (cj, ci) in cells:
                y.append(0.5 * (cells[(ci, cj)] - cells[(cj, ci)]))
                x1.append(np.log(norms[cj]) - np.log(norms[ci]))
                x2.append(s[ci] - s[cj])
    return np.array(y), np.array(x1), np.array(x2)


def fig_asymmetry_g1():
    with open(EVAL / "analysis/registered_reads.json") as f:
        reads = json.load(f)
    fr = [reads["h_asymmetry_raw"][b]["raw_antisym_fraction"] for b in BEH_ORDER]
    corr = reads["h_asymmetry_marker_question_split"]["corrected_antisym_fraction"]
    y, x1, x2 = _g1_points()

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.32)

    axA.bar(range(5), fr, color=paper_palette_role("primary"), width=0.6)
    axA.scatter(
        [0],
        [corr],
        marker="D",
        s=42,
        color=paper_palette_role("accent"),
        zorder=5,
        label="question-split corrected",
    )
    axA.axhline(0.10, color="0.4", lw=0.8, ls="--", label="registered 10% bar")
    axA.set_xticks(range(5))
    axA.set_xticklabels([BEH_LABELS[b] for b in BEH_ORDER], rotation=25, ha="right", fontsize=8)
    axA.set_ylabel("Antisymmetric fraction of\noff-diagonal variance (16×16 block)")
    axA.legend(fontsize=7.5, loc="upper left")
    axA.set_title("Transfer is directional", loc="left", fontweight="semibold", fontsize=10.5)

    axB.scatter(x1, y, s=18, alpha=0.7, color=paper_palette_role("primary"))
    b1 = np.polyfit(x1, y, 1)
    xs = np.linspace(x1.min(), x1.max(), 20)
    axB.plot(xs, np.polyval(b1, xs), color=paper_palette_role("accent"), lw=1.4)
    axB.plot(xs, xs, color="0.5", lw=1.0, ls="--", label="rank-1 prediction (slope 1)")
    axB.set_xlabel("Δ log context norm  (log‖v_j‖ − log‖v_i‖)")
    axB.set_ylabel("Antisymmetric leak ΔG_anti (nats)")
    axB.legend(fontsize=7.5, loc="lower right")
    axB.set_title(
        "Context norms do not predict it", loc="left", fontweight="semibold", fontsize=10.5
    )

    axC.scatter(x2, y, s=18, alpha=0.7, color=paper_palette_role("primary"))
    b2 = np.polyfit(x2, y, 1)
    xs2 = np.linspace(x2.min(), x2.max(), 20)
    axC.plot(xs2, np.polyval(b2, xs2), color=paper_palette_role("accent"), lw=1.4)
    axC.set_xlabel("Implant-strength difference  (s_i − s_j, nats)")
    axC.set_ylabel("Antisymmetric leak ΔG_anti (nats)")
    axC.set_title(
        "Implant-strength difference does", loc="left", fontweight="semibold", fontsize=10.5
    )

    fig.text(
        0.07,
        0.94,
        "Marker row, 55 context pairs (quarantine-masked, flagged cells excluded): "
        "raw scatters behind the registered joint regression",
        fontsize=11.5,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_537/asymmetry_g1", dir=FIGDIR)
    plt.close(fig)


def fig_g2_parallelism():
    with open(EVAL / "analysis/g2_parallelism.json") as f:
        g2 = json.load(f)
    layers = ["layer_6", "layer_14", "layer_22", "layer_27"]
    lab = ["Layer 6", "Layer 14", "Layer 22", "Layer 27"]
    par = [g2["layers"][L]["parallelism_mean_pairwise_cos"] for L in layers]
    flo = [g2["layers"][L]["floor_cross_adapter_mean"] for L in layers]
    cei = [g2["layers"][L]["ceiling_split_half_mean"] for L in layers]
    nul = [g2["layers"][L]["anisotropy_null_mean"] for L in layers]
    rhos = [s["spearman_norm_vs_proj"] for s in g2["scaling_l22_shared16"]["per_adapter"]]

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), constrained_layout=False, gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.22, wspace=0.42)
    x = np.arange(4)
    w = 0.2
    axL.bar(
        x - 1.5 * w,
        par,
        width=w,
        color=paper_palette_role("primary"),
        label="trained Δh, across contexts",
    )
    axL.bar(
        x - 0.5 * w,
        flo,
        width=w,
        color=paper_palette_role("baseline"),
        label="floor: across adapters (common mode)",
    )
    axL.bar(
        x + 0.5 * w,
        cei,
        width=w,
        color=paper_palette_role("neutral"),
        label="ceiling: split-half (same cell)",
    )
    axL.bar(
        x + 1.5 * w,
        nul,
        width=w,
        color=paper_palette_role("control"),
        label="null: base-side anisotropy",
    )
    axL.set_xticks(x)
    axL.set_xticklabels(lab)
    axL.set_ylabel("Mean pairwise cosine")
    axL.axhline(0, color="0.4", lw=0.8)
    axL.legend(fontsize=7.5, loc="upper left")
    axL.set_title(
        "One shared direction per adapter, well above common mode",
        loc="left",
        fontweight="semibold",
        fontsize=10.5,
    )

    jit = np.random.default_rng(0).uniform(-0.06, 0.06, len(rhos))
    axR.scatter(jit, rhos, s=26, alpha=0.8, color=paper_palette_role("primary"))
    axR.axhline(0, color="0.4", lw=0.8)
    axR.axhline(
        float(np.median(rhos)), color=paper_palette_role("accent"), lw=1.4, label="median 0.63"
    )
    axR.set_xlim(-0.5, 0.5)
    axR.set_xticks([])
    axR.set_ylabel("Spearman rho")
    axR.set_xlabel("16 marker adapters (L22),\n‖Δh‖ vs projection coeff.")
    axR.legend(fontsize=7.5, loc="lower right")
    axR.set_title("Magnitude scaling: mixed", loc="left", fontweight="semibold", fontsize=10.5)

    fig.text(
        0.08,
        0.94,
        "Activation-delta parallelism at the marker slot (16 adapters × 30 eval contexts)",
        fontsize=11.5,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_537/g2_parallelism", dir=FIGDIR)
    plt.close(fig)


def fig_em_contrastive_nc():
    with open(EVAL / "analysis/em_contrastive_vs_nc.json") as f:
        em = json.load(f)
    ctxs = ["default", "sp_swe", "wc_short_advice", "fmt_code"]
    ctx_lab = [
        "Default assistant",
        "SW-engineer persona",
        "Chat prefix: advice",
        "Code-comment wrap",
    ]
    c_off = [em["per_train_ctx"][c]["contrastive"]["offdiag_mean"] for c in ctxs]
    n_off = [em["per_train_ctx"][c]["non_contrastive"]["offdiag_mean"] for c in ctxs]
    c_n10 = [em["per_train_ctx"][c]["contrastive"]["n_cells_gt_0p10"] for c in ctxs]
    n_n10 = [em["per_train_ctx"][c]["non_contrastive"]["n_cells_gt_0p10"] for c in ctxs]

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.24, wspace=0.26)
    x = np.arange(4)
    w = 0.36
    axL.bar(
        x - w / 2,
        n_off,
        width=w,
        color=paper_palette_role("control"),
        label="Betley-style, no negatives",
    )
    axL.bar(
        x + w / 2,
        c_off,
        width=w,
        color=paper_palette_role("primary"),
        label="contrastive (testbed default)",
    )
    axL.axhline(0, color="0.4", lw=0.8)
    axL.set_xticks(x)
    axL.set_xticklabels(ctx_lab, rotation=20, ha="right", fontsize=8.5)
    axL.set_ylabel("Mean off-diagonal G\n(Δ P(misaligned) at 29 other contexts)")
    axL.legend(fontsize=8)
    axL.set_title("How much harmful advice leaks", loc="left", fontweight="semibold", fontsize=10.5)

    axR.bar(
        x - w / 2,
        n_n10,
        width=w,
        color=paper_palette_role("control"),
        label="Betley-style, no negatives",
    )
    axR.bar(
        x + w / 2,
        c_n10,
        width=w,
        color=paper_palette_role("primary"),
        label="contrastive (testbed default)",
    )
    axR.set_xticks(x)
    axR.set_xticklabels(ctx_lab, rotation=20, ha="right", fontsize=8.5)
    axR.set_ylabel("# of 29 other contexts with\nΔ P(misaligned) > 0.10")
    axR.legend(fontsize=8)
    axR.set_title("How many contexts it reaches", loc="left", fontweight="semibold", fontsize=10.5)

    fig.text(
        0.08,
        0.94,
        "Emergent misalignment: training without contrastive negatives leaks to nearly every context",
        fontsize=11.5,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_537/em_contrastive_nc", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    fig_behavior_dependence()
    fig_asymmetry_g1()
    fig_g2_parallelism()
    fig_em_contrastive_nc()
    print("figures part 2 written")
