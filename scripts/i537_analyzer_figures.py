"""Issue #537 analyzer figures (round 1).

Generates from the shipped G tensor + leaderboard + registered reads:
  1. g_heatmap_5behaviors  -- hero candidate: 16x30 G heatmap per behavior
  2. g_heatmap_marker      -- marker-row-only heatmap (legible single panel)
  3. leaderboard_marker    -- 26-row metric leaderboard (Spearman + oof R^2)
  4. inoculation_dotplot   -- instruction-trained vs default-trained off-diag G
  5. marker_diag_band      -- diagonal implant strength vs the [5,12] nat band

Saves into the worktree's figures/issue_537/ (committed on the issue-537
branch; body URLs pin that commit SHA).
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

CTX_LABELS = {
    "sp_swe": "SW-engineer persona",
    "sp_doctor": "Doctor persona",
    "sp_ph1": "PersonaHub persona 1",
    "sp_ph2": "PersonaHub persona 2",
    "wc_short_code": "Chat prefix: short, coding",
    "wc_short_advice": "Chat prefix: short, advice",
    "wc_long_write": "Chat prefix: long, writing",
    "icl_k2": "2 worked examples (ICL)",
    "icl_k8": "8 worked examples (ICL)",
    "reph_imp": "Imperative phrasing",
    "reph_polite": "Polite phrasing",
    "reph_casual": "Casual phrasing",
    "fmt_json": "JSON-output instruction",
    "fmt_code": "Code-comment wrap",
    "default": "Default assistant",
    "sp_teacher_ho": "Teacher persona [held out]",
    "sp_ph3_ho": "PersonaHub persona 3 [held out]",
    "wc_short_ho": "Chat prefix: short [held out]",
    "wc_long_ho": "Chat prefix: long [held out]",
    "wc_xlong_ho": "Chat prefix 4-5k tok [held out]",
    "wc_xxlong_ho": "Chat prefix 7-9k tok [held out]",
    "icl_k4_ho": "4 worked examples [held out]",
    "reph_formal_ho": "Formal phrasing [held out]",
    "reph_socratic_ho": "Socratic phrasing [held out]",
    "fmt_mdtable_ho": "Markdown-table instr. [held out]",
    "binst_marker": "Instr: “end with ※”",
    "binst_fact": "Instr: “you believe [fact]”",
    "binst_refusal": "Instr: “refuse everything”",
    "binst_sycophancy": "Instr: “you are sycophantic”",
    "binst_em": "Instr: “you are malicious”",
}

BEHAVIOR_LABELS = {
    "marker": "Marker tic ※  (Δ log-prob, nats)",
    "fact": "Taught fact  (Δ stated-fact rate)",
    "refusal": "Blanket refusal  (Δ refusal rate)",
    "sycophancy": "Sycophancy  (Δ agreement rate)",
    "em": "Harmful advice / EM  (Δ P(misaligned))",
}

METRIC_LABELS = {
    "rbf_mmd2": "RBF MMD² (activation clouds)",
    "bures_w2": "Bures-Wasserstein distance",
    "euclidean": "Euclidean distance of context means",
    "delta_spectrum_mean_norm": "Mean-shift norm (Δ-spectrum)",
    "js_first_token": "First-token JS divergence",
    "mahalanobis_pooled": "Mahalanobis (pooled covariance)",
    "mahalanobis_pair": "Mahalanobis (pairwise covariance)",
    "centroid_cosine": "Cosine distance of context means",
    "kl_first_token_rev": "First-token KL (eval→train)",
    "kl_first_token_fwd": "First-token KL (train→eval)",
    "rank1_proj_raw": "Rank-1 projection (raw)",
    "rank1_proj_whitened": "Rank-1 projection (whitened)",
    "norm_ratio": "Context-norm ratio",
    "content_free": "Content-free probe distance",
    "c2st": "Classifier two-sample test",
    "cos_to_trained_midpoint": "Cosine to trained midpoint [null anchor]",
    "js_to_assistant": "JS to default assistant [null anchor]",
    "cos_to_assistant": "Cosine to default assistant [null anchor]",
    "js_to_neutral": "JS to neutral context [null anchor]",
    "cos_to_neutral": "Cosine to neutral context [null anchor]",
    "delta_spectrum_effective_dim": "Effective dimension (Δ-spectrum)",
    "delta_spectrum_coherence": "Coherence (Δ-spectrum)",
    "neg_panel_prox": "Proximity to negative panel",
    "base_prior_bystander": "Base behavior rate at eval context",
    "gauss_kl_act": "Gaussian KL in activation space",
    "centroid_cosine[end_of_system,L02,centered]": "Cosine of means (early layer, centered)",
}


def load_tensor():
    d = np.load(EVAL / "G_tensor/G_tensor.npz", allow_pickle=True)
    behaviors = [str(b) for b in d["behaviors"]]
    train_cids = [[str(c) for c in row] for row in d["train_cids"]]
    eval_cids = [str(c) for c in d["eval_cids"][0]]
    return d, behaviors, train_cids, eval_cids


def fig_heatmaps():
    d, behaviors, train_cids, eval_cids = load_tensor()
    G = d["G"][..., 0]
    IF = d["implant_failed"][..., 0]

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(5, 1, figsize=(13.5, 24), constrained_layout=False)
    fig.subplots_adjust(left=0.20, right=0.93, top=0.965, bottom=0.045, hspace=0.42)
    for bi, b in enumerate(behaviors):
        ax = axes[bi]
        M = G[bi].copy()
        if b == "marker":
            vmax, vmin, cmap = 8.0, -8.0, "RdBu_r"
        else:
            vmax, vmin, cmap = 1.0, -1.0, "RdBu_r"
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        # hatch implant-failed rows
        for i in range(16):
            if IF[bi, i, :].any():
                ax.add_patch(
                    plt.Rectangle(
                        (-0.5, i - 0.5), 30, 1, fill=False, hatch="///", edgecolor="0.35", lw=0
                    )
                )
        # diagonal boxes (shared instances + own binst)
        for i, ci in enumerate(train_cids[bi]):
            if ci in eval_cids:
                j = eval_cids.index(ci)
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.1)
                )
        # family separators: shared-train block ends after col 14, held-outs after col 24
        for x in (14.5, 24.5):
            ax.axvline(x, color="black", lw=0.8)
        ax.set_xticks(range(30))
        ax.set_yticks(range(16))
        ax.set_yticklabels([CTX_LABELS[c] for c in train_cids[bi]], fontsize=7)
        if bi == 4:
            ax.set_xticklabels(
                [CTX_LABELS[c] for c in eval_cids], fontsize=7, rotation=30, ha="right"
            )
            ax.set_xlabel("Eval context (where the behavior is measured)")
        else:
            ax.set_xticklabels([])
        ax.set_ylabel("Train context")
        ax.set_title(BEHAVIOR_LABELS[b], loc="left", fontweight="semibold", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.ax.tick_params(labelsize=7)
        ax.grid(False)
    fig.text(
        0.20,
        0.985,
        "The generalization tensor: train a behavior under one context, measure it under 30",
        fontsize=14,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_537/g_heatmap_5behaviors", dir=FIGDIR)
    plt.close(fig)


def fig_marker_heatmap():
    d, behaviors, train_cids, eval_cids = load_tensor()
    bi = behaviors.index("marker")
    G = d["G"][bi, :, :, 0]
    IF = d["implant_failed"][bi, :, :, 0]
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.86, bottom=0.24)
    im = ax.imshow(G, aspect="auto", cmap="RdBu_r", vmin=-8, vmax=8)
    for i in range(16):
        if IF[i, :].any():
            ax.add_patch(
                plt.Rectangle(
                    (-0.5, i - 0.5), 30, 1, fill=False, hatch="///", edgecolor="0.35", lw=0
                )
            )
    for i, ci in enumerate(train_cids[bi]):
        if ci in eval_cids:
            j = eval_cids.index(ci)
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.2)
            )
    for x in (14.5, 24.5):
        ax.axvline(x, color="black", lw=0.8)
    ax.set_xticks(range(30))
    ax.set_xticklabels([CTX_LABELS[c] for c in eval_cids], fontsize=7.5, rotation=30, ha="right")
    ax.set_yticks(range(16))
    ax.set_yticklabels([CTX_LABELS[c] for c in train_cids[bi]], fontsize=8)
    ax.set_xlabel("Eval context")
    ax.set_ylabel("Train context")
    ax.set_title(
        "Marker row of the G tensor: Δ log-prob of ※ at the end-of-answer slot (nats)\n"
        "boxes = train-context diagonal; hatched row = implant failed; "
        "right block = behavior instructions",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("G (trained - base, nats)", fontsize=9)
    ax.grid(False)
    savefig_paper(fig, "issue_537/g_heatmap_marker", dir=FIGDIR)
    plt.close(fig)


def fig_leaderboard():
    with open(EVAL / "baselines/baseline_scores.json") as f:
        bs = json.load(f)
    rows = []
    for k, v in bs["scores"].items():
        mid = k.split(":", 1)[1]
        rows.append(
            (
                METRIC_LABELS.get(mid, mid),
                v["spearman"],
                v["oof_r2"],
                v["bootstrap"]["ci_lo"],
                v["bootstrap"]["ci_hi"],
            )
        )
    rows.sort(key=lambda r: r[2])
    labels = [r[0] for r in rows]
    rho = np.array([r[1] for r in rows])
    r2 = np.array([r[2] for r in rows])
    lo = np.array([r[3] for r in rows])
    hi = np.array([r[4] for r in rows])

    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 7.6), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.90, bottom=0.09, wspace=0.08)
    y = np.arange(len(rows))
    null_mask = np.array(["null anchor" in lbl for lbl in labels])
    colors = [
        paper_palette_role("neutral") if nm else paper_palette_role("primary") for nm in null_mask
    ]
    axL.errorbar(
        rho,
        y,
        xerr=[rho - lo, hi - rho],
        fmt="o",
        ms=4.5,
        color=paper_palette_role("primary"),
        ecolor="0.6",
        elinewidth=1,
        capsize=2,
        zorder=3,
    )
    axL.scatter(rho[null_mask], y[null_mask], s=22, color=paper_palette_role("neutral"), zorder=4)
    axL.axvline(0, color="0.4", lw=0.8)
    axL.set_yticks(y)
    axL.set_yticklabels(labels, fontsize=8)
    axL.set_xlabel("Spearman rho vs marker G (95% bootstrap CI)\nmore negative = better predictor")
    axR.barh(y, r2, color=colors, height=0.62)
    axR.axvline(0, color="0.4", lw=0.8)
    axR.set_xlabel("Out-of-fold R² (leave-two-contexts-out CV)\nhigher = better")
    fig.text(
        0.32,
        0.955,
        "Marker-row leaderboard: 26 registered metrics, quarantine-masked (n = 193 cells)",
        fontsize=12.5,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_537/leaderboard_marker", dir=FIGDIR)
    plt.close(fig)


def fig_inoculation():
    with open(EVAL / "analysis/registered_reads.json") as f:
        reads = json.load(f)
    inoc = reads["h_inoculation"]
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.18, wspace=0.25)
    # Left: marker (nats)
    m = inoc["marker"]
    axL.bar(
        [0, 1],
        [m["default_offdiag_mean"], m["binst_offdiag_mean"]],
        color=[paper_palette_role("baseline"), paper_palette_role("primary")],
        width=0.55,
    )
    axL.set_xticks([0, 1])
    axL.set_xticklabels(
        ["Trained under\ndefault assistant", "Trained under\n“end with ※” instruction"]
    )
    axL.set_ylabel("Mean off-diagonal G (nats)")
    axL.set_title("Marker tic ※", loc="left", fontweight="semibold", fontsize=11)
    axL.axhline(0, color="0.4", lw=0.8)
    # Right: rate behaviors
    rate_bs = ["fact", "sycophancy", "em"]
    x = np.arange(len(rate_bs))
    w = 0.36
    dvals = [inoc[b]["default_offdiag_mean"] for b in rate_bs]
    bvals = [inoc[b]["binst_offdiag_mean"] for b in rate_bs]
    axR.bar(
        x - w / 2,
        dvals,
        width=w,
        color=paper_palette_role("baseline"),
        label="Trained under default assistant",
    )
    axR.bar(
        x + w / 2,
        bvals,
        width=w,
        color=paper_palette_role("primary"),
        label="Trained under its own instruction",
    )
    axR.set_xticks(x)
    axR.set_xticklabels(["Taught fact", "Sycophancy", "Harmful advice"])
    axR.set_ylabel("Mean off-diagonal G (rate delta)")
    axR.set_title("Rate behaviors", loc="left", fontweight="semibold", fontsize=11)
    axR.axhline(0, color="0.4", lw=0.8)
    axR.legend(fontsize=8, loc="upper right")
    fig.text(
        0.09,
        0.95,
        "Inoculation: training under an explicit behavior instruction "
        "keeps the behavior out of every other context",
        fontsize=12,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_537/inoculation_dotplot", dir=FIGDIR)
    plt.close(fig)


def fig_marker_diag_band():
    _d, behaviors, train_cids, _eval_cids = load_tensor()
    with open(EVAL / "G_tensor/G_meta.json") as f:
        meta = json.load(f)
    pc = meta["per_cell"]
    bi = behaviors.index("marker")
    tc = train_cids[bi]
    diag = []
    stops = []
    for ci in tc:
        cell = pc[f"marker/{ci}__{ci}"]
        diag.append(cell["g"])
        stops.append(cell.get("stop_step"))
    set_paper_style("blog")
    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.32)
    x = np.arange(16)
    ax.axhspan(5, 12, color=paper_palette_role("accent"), alpha=0.15, zorder=0)
    colors = []
    for g in diag:
        if g < 4:
            colors.append(paper_palette_role("control"))
        elif g > 12:
            colors.append(paper_palette_role("accent"))
        else:
            colors.append(paper_palette_role("primary"))
    ax.bar(x, diag, color=colors, width=0.62)
    for xi, (g, s) in enumerate(zip(diag, stops, strict=True)):
        ax.text(xi, g + 0.4, f"step {s}", ha="center", fontsize=7, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([CTX_LABELS[c] for c in tc], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Diagonal implant strength\nG = Δ log P(※), nats")
    ax.text(0.3, 11.0, "target band [5, 12] nat", fontsize=8.5, color="0.3")
    ax.set_title(
        "Band-stopped marker training: 14/16 train cells land in band; "
        "code-comment wrap falls short (3.97);\n"
        "the “end with ※” instruction cell overshoots to 25.2 nat (saturated)",
        loc="left",
        fontsize=10.5,
        fontweight="semibold",
    )
    savefig_paper(fig, "issue_537/marker_diag_band", dir=FIGDIR)
    plt.close(fig)


if __name__ == "__main__":
    fig_heatmaps()
    fig_marker_heatmap()
    fig_leaderboard()
    fig_inoculation()
    fig_marker_diag_band()
    print("figures written")
