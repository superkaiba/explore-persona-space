#!/usr/bin/env python3
"""Issue #667 clean-result figures (blog style).

Reads the per-assumption JSONs (eval_results/issue_667/A3_*.json) + the per-cell
store + #537 G_meta + #658 sigma_c, builds the clean-result figures:

  fig_a39_a310_forest   hero: base gate g0 / oracle g+ vs realized activation gate, per behavior
  fig_a39_scatter       low-level per-cell scatter behind the headline (g0 vs ĝ^real)
  fig_gate_vs_behavior  KEY CAVEAT: gate predicts activation gate strongly, behavioral leakage G weakly
  fig_a38_rankone       A3.8 stacked-ΔV σ1²/Σσ² per behavior vs chance + per-source points
  fig_a37_write         A3.7 cos(ŵ,δ) pos/contra vs null per behavior + per-source points
  fig_a36_forest        A3.6 partial-Spearman(change|base) per behavior vs null
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    _ols_residual as ols_residual,
)
from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    _rankdata as rankdata,
)
from explore_persona_space.analysis.issue667.gate_chain import (  # noqa: E402
    default_lambda,
    readout_projection,
    spearman_rho,
)
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

D = ROOT / "eval_results" / "issue_667"
TENS = D / "analysis_tensors"
G_META = json.loads((ROOT / "eval_results/issue_537/G_tensor/G_meta.json").read_text())["per_cell"]

BEH_LABEL = {
    "em": "Emergent misalignment",
    "sycophancy": "Sycophancy",
    "fact": "Taught fact",
    "marker": "Marker (saturated, supplement)",
}
HEADLINE = ["em", "sycophancy", "fact"]
ALL_BEH = ["em", "sycophancy", "fact", "marker"]


def _load(f):
    return json.loads((D / f).read_text())


def _sigma_inv(layer=14):
    from huggingface_hub import hf_hub_download

    sig = torch.load(
        hf_hub_download(
            "superkaiba1/explore-persona-space-data",
            "issue658_theory_assumptions/store/sigma_c.pt",
            repo_type="dataset",
        ),
        weights_only=False,
        map_location="cpu",
    )
    cap = list(sig["capture_layers"])
    S = sig["sigma_c"][cap.index(layer)].to(torch.float64)
    lam = default_lambda(S)
    return torch.linalg.inv(S + lam * torch.eye(S.shape[0], dtype=torch.float64))


def _g0(sinv, c_C, c_Cp):
    zc = sinv @ torch.from_numpy(c_C).double()
    return float((zc @ torch.from_numpy(c_Cp).double()) / (zc @ torch.from_numpy(c_C).double()))


def _realized_gate(src_npz, tgt_npz):
    w = src_npz["v_plus"].astype(np.float64) - src_npz["v0"].astype(np.float64)
    dv = tgt_npz["v_plus"].astype(np.float64) - tgt_npz["v0"].astype(np.float64)
    return float((w @ dv) / (w @ w))


def gather_scatter(sinv):
    """{behavior: (g0_pred[], g_real[], g_behavioral[], labels[])} off-diagonal cells."""
    out = {}
    for beh in ALL_BEH:
        base = TENS / beh
        diag = {}
        for s in os.listdir(base):
            src = s.rsplit("_seed", 1)[0]
            diag[src] = dict(np.load(base / s / f"{src}_L14.npz", allow_pickle=True))
        g0s, greals, gbeh, labels = [], [], [], []
        for s in os.listdir(base):
            src = s.rsplit("_seed", 1)[0]
            for f in os.listdir(base / s):
                if not f.endswith("_L14.npz"):
                    continue
                tgt = f[: -len("_L14.npz")]
                if tgt == src:
                    continue
                d = dict(np.load(base / s / f, allow_pickle=True))
                g0s.append(_g0(sinv, diag[src]["c_C"], d["c_Cp"]))
                greals.append(_realized_gate(diag[src], d))
                cell = G_META.get(f"{beh}/{src}__{tgt}")
                gbeh.append(cell["g"] if cell else np.nan)
                labels.append(f"{src}->{tgt}")
        out[beh] = (np.array(g0s), np.array(greals), np.array(gbeh), labels)
    return out


def fig_a39_a310_forest():
    a39 = _load("A3_9_key_query_gate.json")["by_behavior"]
    a310 = _load("A3_10_base_gate_validity.json")["by_behavior"]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    behs = HEADLINE + ["marker"]
    y = np.arange(len(behs))[::-1]
    c_g0 = paper_palette_role("primary")
    c_or = paper_palette_role("baseline")
    c_cos = paper_palette_role("control")
    for i, b in enumerate(behs):
        r9 = a39[b]
        bb = r9["boxed_primary_clustered_bootstrap"]
        yy = y[i]
        # base gate g0 (== boxed c_C whitened) vs realized
        ax.plot(
            [bb["ci_lo"], bb["ci_hi"]],
            [yy + 0.16, yy + 0.16],
            color=c_g0,
            lw=2.4,
            solid_capstyle="round",
        )
        ax.scatter(
            [r9["boxed_primary_spearman"]],
            [yy + 0.16],
            color=c_g0,
            s=70,
            zorder=5,
            label="Base gate g0 → realized gate" if i == 0 else None,
        )
        # oracle g+ vs realized
        r10 = a310[b]
        ob = r10.get("oracle_gplus_vs_realized_clustered_bootstrap", {})
        if ob:
            ax.plot(
                [ob["ci_lo"], ob["ci_hi"]],
                [yy - 0.16, yy - 0.16],
                color=c_or,
                lw=2.4,
                solid_capstyle="round",
            )
            ax.scatter(
                [r10["oracle_gplus_vs_realized_spearman"]],
                [yy - 0.16],
                color=c_or,
                marker="D",
                s=55,
                zorder=5,
                label="Post-FT oracle gate g+ → realized gate" if i == 0 else None,
            )
        # true-cosine baseline (open marker)
        ax.scatter(
            [r9["true_cosine_baseline_spearman"]],
            [yy],
            facecolors="none",
            edgecolors=c_cos,
            s=55,
            linewidths=1.6,
            zorder=4,
            label="Plain cosine baseline" if i == 0 else None,
        )
    null_hi = max(a39[b]["boxed_primary_shuffled_null_hi"] for b in behs)
    ax.axvspan(-null_hi, null_hi, color="0.85", alpha=0.6, zorder=0)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([BEH_LABEL[b] for b in behs])
    ax.set_xlabel("Spearman ρ  (predicted gate vs realized activation gate)")
    ax.set_xlim(-0.18, 0.82)
    # all points sit at ρ > 0.25, so the left half is empty — park the legend there.
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=8, frameon=False)
    set_title_subtitle(
        ax,
        "The base-model gate predicts the realized activation gate as well as the post-FT oracle",
        "Per-behavior Spearman ρ, 95% family-clustered bootstrap CI; grey band = shuffled-key null. n=464 cells/behavior.",
    )
    savefig_paper(fig, "issue_667/fig_a39_a310_forest", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def fig_a39_scatter(scatter):
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=False)
    a39 = _load("A3_9_key_query_gate.json")["by_behavior"]
    c = paper_palette_role("primary")
    for ax, b in zip(axes, HEADLINE, strict=True):
        g0s, greals, _gb, _lab = scatter[b]
        ax.scatter(g0s, greals, s=12, alpha=0.45, color=c, edgecolors="none")
        rho = a39[b]["boxed_primary_spearman"]
        ax.set_title(f"{BEH_LABEL[b]}\nρ = {rho:+.2f}", fontsize=9.5)
        ax.set_xlabel("Base whitened gate g0(C′)")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
    axes[0].set_ylabel("Realized activation gate ĝʳᵉᵃˡ(C′)")
    fig.suptitle(
        "Per-cell base gate vs realized activation gate (the data behind the ρ)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "issue_667/fig_a39_scatter", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def fig_gate_vs_behavior(scatter):
    """KEY CAVEAT: gate -> activation gate strong, gate -> behavioral leakage G weak."""
    a39 = _load("A3_9_key_query_gate.json")["by_behavior"]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    behs = HEADLINE
    x = np.arange(len(behs))
    act = [a39[b]["boxed_primary_spearman"] for b in behs]
    beh_rho = []
    for b in behs:
        g0s, _gr, gb, _l = scatter[b]
        mask = np.isfinite(gb)
        beh_rho.append(spearman_rho(g0s[mask], gb[mask]))
    w = 0.36
    ax.bar(
        x - w / 2,
        act,
        w,
        color=paper_palette_role("primary"),
        label="vs realized activation gate ĝʳᵉᵃˡ (mechanism)",
    )
    ax.bar(
        x + w / 2,
        beh_rho,
        w,
        color=paper_palette_role("accent"),
        label="vs measured behavioral leakage G (behavior)",
    )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b] for b in behs])
    ax.set_ylabel("Spearman ρ  (base gate vs target)")
    ax.set_ylim(0, 0.82)
    # bars peak ~0.59; the band above them (right two groups) is open — put the
    # 2-row legend there, anchored to the upper-right of the data area.
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=8.5, frameon=False, ncol=1)
    set_title_subtitle(
        ax,
        "Activation-space gate relation observed; behavioral translation remains partial",
        "Base gate predicts the activation gate strongly; its reach to the measured leakage matrix G is weak for EM/sycophancy, moderate for fact. n=464.",
    )
    # set_title_subtitle + blog constrained_layout collapses the axes (memory:
    # set_title_subtitle_breaks_subplot_grids) — switch to explicit margins.
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.84)
    savefig_paper(fig, "issue_667/fig_gate_vs_behavior", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def fig_gate_vs_behavior_scatter(scatter):
    """Per-cell data behind the behavioral bar of fig_gate_vs_behavior:
    base gate g0 vs measured behavioral leakage G, one point per off-diagonal cell."""
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=False)
    c = paper_palette_role("accent")
    for ax, b in zip(axes, HEADLINE, strict=True):
        g0s, _gr, gb, _lab = scatter[b]
        mask = np.isfinite(gb)
        g0m, gbm = g0s[mask], gb[mask]
        ax.scatter(g0m, gbm, s=12, alpha=0.45, color=c, edgecolors="none")
        rho = spearman_rho(g0m, gbm)
        ax.set_title(f"{BEH_LABEL[b]}\nρ = {rho:+.2f}", fontsize=9.5)
        ax.set_xlabel("Base whitened gate g0(C′)")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
    axes[0].set_ylabel("Measured behavioral leakage G")
    fig.suptitle(
        "Per-cell base gate vs measured behavioral leakage (the data behind the weak G bar)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "issue_667/fig_gate_vs_behavior_scatter", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def fig_a38_rankone():
    a38 = _load("A3_8_rank_one.json")["by_behavior"]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    behs = ALL_BEH
    x = np.arange(len(behs))
    c = paper_palette_role("primary")
    for i, b in enumerate(behs):
        ps = a38[b]["per_source"]
        vals = np.array([p["sigma1_sq_frac"] for p in ps])
        jit = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.28
        ax.scatter(x[i] + jit, vals, s=16, alpha=0.5, color=c, edgecolors="none")
        ax.scatter(
            [x[i]], [np.median(vals)], marker="_", s=900, color="0.15", zorder=5, linewidths=2.2
        )
    chance = a38["em"]["per_source"][0]["chance_sigma1_frac"]
    ax.axhline(
        chance,
        color=paper_palette_role("control"),
        lw=1.4,
        ls="--",
        label=f"chance ≈ 1/29 = {chance:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b].split(" (")[0] for b in behs], rotation=12, ha="right")
    ax.set_ylabel("Top-singular variance fraction  σ₁² / Σσ²")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    set_title_subtitle(
        ax,
        "Each source's off-target updates are dominantly a single shared direction",
        "Per-source stacked-ΔV top-singular fraction (one dot per source, bar = median); ~24× chance. cos(top dir, source write) median 0.85–0.93.",
    )
    savefig_paper(fig, "issue_667/fig_a38_rankone", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def fig_a37_write():
    a37 = _load("A3_7_source_write.json")["by_behavior"]
    set_paper_style("blog")
    # taller canvas so blog constrained_layout reserves room for the rotated
    # x-tick behavior labels + the long y-axis label (no clipping at the bottom).
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    behs = HEADLINE + ["marker"]
    x = np.arange(len(behs))
    w = 0.27
    c_pos = paper_palette_role("primary")
    c_con = paper_palette_role("baseline")
    c_null = paper_palette_role("control")
    pos = [a37[b]["mean_cos_pos"] for b in behs]
    con = [a37[b]["mean_cos_contra"] for b in behs]
    nul = [a37[b]["mean_cos_null"] for b in behs]
    ax.bar(x - w, pos, w, color=c_pos, label="cos(write, positive-only target δᵖᵒˢ)")
    ax.bar(x, con, w, color=c_con, label="cos(write, contrastive target δᶜᵒⁿᵗʳᵃ)")
    ax.bar(x + w, nul, w, color=c_null, label="shuffled-δ null")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([BEH_LABEL[b].split(" (")[0] for b in behs], rotation=12, ha="right")
    ax.set_ylabel("Mean cosine(source write, displacement target)")
    ax.set_ylim(-0.26, 0.21)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    # Annotate per-behavior mean frac_ctx (source-vs-negative context offset as a
    # fraction of the contrastive displacement): EM's ~1.0 means the offset IS the
    # whole displacement, which is why EM's pos/contra cosines can diverge.
    for xi, b in zip(x, behs, strict=True):
        fc = a37[b]["mean_frac_ctx"]
        ax.annotate(
            f"frac_ctx\n{fc:.2f}",
            (xi, 0.185),
            ha="center",
            va="top",
            fontsize=7,
            color="0.35",
        )
    set_title_subtitle(
        ax,
        "The realized write does not point toward the training-data target",
        "Mean cos(write, δ) per behavior; positive-only ≈ contrastive at the mean. frac_ctx = context offset as a fraction of the contrastive displacement.",
    )
    savefig_paper(fig, "issue_667/fig_a37_write", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def gather_a36_residuals():
    """{behavior: (x_resid[], y_resid[])} — the per-cell partial residuals A3.6 ranks.

    Mirrors ``issue667_analysis.run_a36`` EXACTLY: x = r_B·Δv(C′), y = E+−E0 (g),
    z = E0 (base_rate); rank all three, OLS-residualize the ranks of x and y on the
    rank of z. The partial Spearman is the Pearson correlation of (x_resid, y_resid),
    so plotting them IS the per-unit data the −0.35 / −0.03 / −0.41 partial ρ reduces.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import issue667_analysis as ana

    g_meta = ana.load_g_meta()
    out = {}
    for beh in HEADLINE:
        cells = ana.load_cells(TENS, beh, 14)
        r_b = ana.load_r_b(beh, 14)
        if r_b is None:  # fact: re-extracted r_b lives in the store cells
            r_b = ana._fact_rb_from_store(cells)
        xs, ys, zs = [], [], []
        for (source, target), data in cells.items():
            if source == target:
                continue  # off-diagonal targets only (the CHANGE read)
            gc = ana.g_cell(g_meta, beh, source, target)
            if gc is None:
                continue
            delta_v = data["v_plus"].astype(np.float64) - data["v0"].astype(np.float64)
            xs.append(readout_projection(r_b, delta_v))
            ys.append(float(gc["g"]))
            zs.append(float(gc["base_rate"]))
        x, y, z = np.array(xs), np.array(ys), np.array(zs)
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
        out[beh] = (ols_residual(rx, rz), ols_residual(ry, rz))
    return out


def fig_a36_scatter():
    """Low-level per-cell partial-residual scatter behind the A3.6 partial ρ."""
    resid = gather_a36_residuals()
    a36 = _load("A3_6_readout_stability.json")["by_behavior"]
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=False)
    c = paper_palette_role("primary")
    for ax, b in zip(axes, HEADLINE, strict=True):
        xr, yr = resid[b]
        ax.scatter(xr, yr, s=12, alpha=0.45, color=c, edgecolors="none")
        rho = a36[b]["partial_spearman_change_given_base"]
        ax.set_title(f"{BEH_LABEL[b]}\nρ = {rho:+.2f}", fontsize=9.5)
        ax.set_xlabel("Read-out · Δv  (base-rate residual)")
        ax.axhline(0, color="0.7", lw=0.6)
        ax.axvline(0, color="0.7", lw=0.6)
    axes[0].set_ylabel("Δbehavior  (base-rate residual)")
    fig.suptitle(
        "Per-cell partial residuals behind the A3.6 partial ρ (read-out·Δv vs Δbehavior | base)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    savefig_paper(fig, "issue_667/fig_a36_scatter", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def fig_a36_forest():
    a36 = _load("A3_6_readout_stability.json")["by_behavior"]
    set_paper_style("blog")
    # taller canvas so blog constrained_layout reserves room for the left y-tick
    # behavior labels + the bottom x-axis label (no clipping). The x-label and
    # subtitle are kept short so constrained_layout, which centers them under the
    # narrow forest axes, does not overflow the figure edges (round-2 clip fix).
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    behs = HEADLINE
    y = np.arange(len(behs))[::-1]
    c = paper_palette_role("primary")
    for i, b in enumerate(behs):
        r = a36[b]
        pb = r["partial_clustered_bootstrap"]
        ax.plot([pb["ci_lo"], pb["ci_hi"]], [y[i], y[i]], color=c, lw=2.6, solid_capstyle="round")
        ax.scatter([r["partial_spearman_change_given_base"]], [y[i]], color=c, s=70, zorder=5)
    null_hi = max(a36[b]["partial_shuffled_null_hi"] for b in behs)
    ax.axvspan(-null_hi, null_hi, color="0.85", alpha=0.6, zorder=0)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([BEH_LABEL[b] for b in behs])
    ax.set_xlabel("Partial Spearman ρ  (read-out · Δv  vs  Δbehavior | base)")
    ax.set_xlim(-0.7, 0.4)
    set_title_subtitle(
        ax,
        "The base read-out does not predict the post-FT behavior change",
        "Partial ρ, base level partialled out; grey band = shuffled-read-out null. EM/fact negative, sycophancy null. n=464.",
    )
    savefig_paper(fig, "issue_667/fig_a36_forest", dir=str(ROOT / "figures") + "/")
    plt.close(fig)


def main():
    sinv = _sigma_inv(14)
    scatter = gather_scatter(sinv)
    fig_a39_a310_forest()
    fig_a39_scatter(scatter)
    fig_gate_vs_behavior(scatter)
    fig_gate_vs_behavior_scatter(scatter)
    fig_a38_rankone()
    fig_a37_write()
    fig_a36_forest()
    fig_a36_scatter()
    print("figures written to figures/issue_667/")


if __name__ == "__main__":
    main()
