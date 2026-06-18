"""Issue #650 paper-plots — DV-1..DV-5 read/write geometry (marker vs sycophancy).

Reads eval_results/issue_650/analysis/dv{1..5}_*.json + eval JSONs, emits the
clean-result figures to figures/issue_650/ (blog style). One figure per finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

A = Path("eval_results/issue_650/analysis")
EVAL = Path("eval_results/issue_650/eval")
SEEDS = (42, 137, 256)


def load(f: str) -> dict:
    return json.loads((A / f).read_text())["cells"]


def cellvals(cells: dict, behavior: str, dose: str, getter) -> list[float]:
    out = []
    for slug, r in cells.items():
        b, d = slug.split("__")[0], slug.split("__")[1]
        if b == behavior and d == dose:
            v = getter(r)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                out.append(float(v))
    return out


def mean_sd(xs: list[float]) -> tuple[float, float]:
    a = np.array(xs)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Figure 1 (HERO): read frozen at init vs write aligned to base/concept.
# Two-panel: (left) DV-1 cos(a_trained,a_init) read-rotation + a∘γ·v_source;
#            (right) DV-3 write intruder obs vs max-matched null.
# ──────────────────────────────────────────────────────────────────────────
def fig_hero() -> None:
    d1 = load("dv1_read_rotation.json")
    d3 = load("dv3_intruder.json")
    groups = [("marker", "low"), ("marker", "high"), ("sycophancy", "low"), ("sycophancy", "high")]
    labels = [
        "Marker\n(low dose)",
        "Marker\n(high dose)",
        "Sycophancy\n(low dose)",
        "Sycophancy\n(high dose)",
    ]
    x = np.arange(len(groups))

    set_paper_style("blog")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Left: read self-similarity to init (cos a_trained, a_init).
    read_m, read_s = [], []
    for b, dose in groups:
        m, s = mean_sd(cellvals(d1, b, dose, lambda r: r["band_mean_cos_a_init"]))
        read_m.append(m)
        read_s.append(s)
    cprim = paper_palette_role("primary")
    cbase = paper_palette_role("baseline")
    axL.bar(
        x,
        read_m,
        yerr=read_s,
        color=cprim,
        capsize=3,
        width=0.62,
        label="read vs its own random init",
    )
    kf = 1.0 / np.sqrt(3584)
    axL.axhline(kf, ls="--", lw=1.2, color=cbase, label=f"random-init floor ({kf:.3f})")
    for xi, m in zip(x, read_m):
        axL.text(
            xi,
            m - 0.06,
            f"{m:.3f}",
            ha="center",
            va="top",
            fontsize=8.5,
            color="white",
            fontweight="bold",
        )
    axL.set_xticks(x)
    axL.set_xticklabels(labels, fontsize=8.5)
    axL.set_ylabel("cosine(read direction, random init)")
    axL.set_ylim(0, 1.05)
    axL.legend(frameon=False, fontsize=8, loc="lower center")
    set_title_subtitle(
        axL,
        "The read direction never moves",
        "rank-1 LoRA read stays ≈ its random init at every dose",
        source=None,
    )

    # Right: DV-3 write obs vs max-matched null.
    obs_m, obs_s, nul = [], [], []
    for b, dose in groups:
        m, s = mean_sd(cellvals(d3, b, dose, lambda r: r["observed"]["write"]["band_max"]))
        nm, _ = mean_sd(cellvals(d3, b, dose, lambda r: r["null"]["write"]["band_p95"]))
        obs_m.append(m)
        obs_s.append(s)
        nul.append(nm)
    caccent = paper_palette_role("accent")
    axR.bar(
        x,
        obs_m,
        yerr=obs_s,
        color=caccent,
        capsize=3,
        width=0.62,
        label="observed max |cos| to base singular vectors",
    )
    # null p95 as a step/line per group
    for xi, nv in zip(x, nul):
        axR.plot(
            [xi - 0.31, xi + 0.31],
            [nv, nv],
            ls="--",
            lw=1.6,
            color=cbase,
            label="max-matched null (p95)" if xi == 0 else None,
        )
    for xi, m in zip(x, obs_m):
        axR.text(xi, m + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)
    axR.set_xticks(x)
    axR.set_xticklabels(labels, fontsize=8.5)
    axR.set_ylabel("write alignment to base weight geometry")
    axR.set_ylim(0, 0.26)
    axR.legend(frameon=False, fontsize=8, loc="upper left")
    set_title_subtitle(
        axR,
        "The write barely touches base weight geometry",
        "marker write sits modestly above null; sycophancy at the intruder floor",
        source=None,
    )

    savefig_paper(fig, "issue_650/hero_read_frozen_write_intruder", dir="figures/")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 2: DV-2 write -> output concept (marker vs sycophancy vs null).
# ──────────────────────────────────────────────────────────────────────────
def fig_dv2() -> None:
    d2 = load("dv2_write_concept.json")
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    behaviors = ["marker", "sycophancy"]
    disp = [
        "Marker\n(write vs unembedding row for ※)",
        "Sycophancy\n(write vs agreement logit-diff direction)",
    ]
    obs_m, obs_s, nul = [], [], []
    for b in behaviors:
        vals = [d2[s]["band_max"] for s in d2 if s.startswith(f"{b}__")]
        nuls = [d2[s]["null_p95"] for s in d2 if s.startswith(f"{b}__")]
        m, s = mean_sd(vals)
        obs_m.append(m)
        obs_s.append(s)
        nul.append(float(np.mean(nuls)))
    x = np.arange(len(behaviors))
    cprim = paper_palette_role("primary")
    cbase = paper_palette_role("baseline")
    ax.bar(
        x,
        obs_m,
        yerr=obs_s,
        color=cprim,
        capsize=3,
        width=0.5,
        label="observed cosine to output concept",
    )
    for xi, nv in zip(x, nul):
        ax.plot(
            [xi - 0.25, xi + 0.25],
            [nv, nv],
            ls="--",
            lw=1.6,
            color=cbase,
            label="frequency-matched null (p95)" if xi == 0 else None,
        )
    for xi, m in zip(x, obs_m):
        ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(disp, fontsize=9)
    ax.set_ylabel("cosine(write direction, output-concept direction)")
    ax.set_ylim(0, 0.92)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    set_title_subtitle(
        ax,
        "The write points at the output concept — strongly for the marker",
        "n=6 cells per behavior; sycophancy alignment is ~2.2× weaker than the marker's",
        source=None,
    )
    savefig_paper(fig, "issue_650/dv2_write_to_output_concept", dir="figures/")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 3: DV-3 per-cell write obs vs max-matched null (strip, both arms).
# ──────────────────────────────────────────────────────────────────────────
def fig_dv3_strip() -> None:
    d3 = load("dv3_intruder.json")
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    groups = [("marker", "low"), ("marker", "high"), ("sycophancy", "low"), ("sycophancy", "high")]
    labels = ["Marker low", "Marker high", "Sycophancy low", "Sycophancy high"]
    cwrite = paper_palette_role("accent")
    cread = paper_palette_role("neutral")
    cbase = paper_palette_role("baseline")
    x = np.arange(len(groups))
    for i, (b, dose) in enumerate(groups):
        wv = cellvals(d3, b, dose, lambda r: r["observed"]["write"]["band_max"])
        rv = cellvals(d3, b, dose, lambda r: r["observed"]["read"]["band_max"])
        ax.scatter(
            np.full(len(wv), i - 0.13),
            wv,
            color=cwrite,
            s=55,
            zorder=3,
            label="WRITE (b_down vs base down_proj)" if i == 0 else None,
        )
        ax.scatter(
            np.full(len(rv), i + 0.13),
            rv,
            color=cread,
            s=55,
            marker="s",
            zorder=3,
            label="READ (a_up vs base up_proj)" if i == 0 else None,
        )
    # single null band (p95 ~ 0.080 across cells)
    nul = float(np.mean([d3[s]["null"]["write"]["band_p95"] for s in d3]))
    ax.axhline(nul, ls="--", lw=1.5, color=cbase, label=f"max-matched null p95 ≈ {nul:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("max |cos| to base singular vectors")
    ax.set_ylim(0, 0.24)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    set_title_subtitle(
        ax,
        "Write vs read against base weight geometry, per cell",
        "each point = one seed; read sits at the null floor everywhere, write rises with marker dose",
        source=None,
    )
    savefig_paper(fig, "issue_650/dv3_intruder_per_cell", dir="figures/")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 4: DV-5 selectivity — learned firing predictor vs plain base geometry.
# ──────────────────────────────────────────────────────────────────────────
def fig_dv5() -> None:
    d5 = load("dv5_selectivity.json")
    cells = {
        s: r for s, r in d5.items() if s.startswith("marker__") and "rho_firing_predictor" in r
    }
    order = sorted(cells)
    disp = [s.replace("marker__", "").replace("__", " ").replace("seed", "seed ") for s in order]
    rho_fire = [abs(cells[s]["rho_firing_predictor"]) for s in order]
    rho_geom = [abs(cells[s]["rho_plain_geometry"]) for s in order]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    x = np.arange(len(order))
    w = 0.38
    cprim = paper_palette_role("primary")
    cbase = paper_palette_role("baseline")
    ax.bar(
        x - w / 2,
        rho_fire,
        w,
        color=cprim,
        label="learned firing predictor (read · bystander context)",
    )
    ax.bar(
        x + w / 2,
        rho_geom,
        w,
        color=cbase,
        label="plain base geometry (context vs source-context cosine)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(disp, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylabel("|Spearman ρ| with bystander leakage")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    set_title_subtitle(
        ax,
        "A learned read does not beat plain base geometry at predicting leakage",
        "16 bystanders/cell; mean Δρ = +0.02 (p = 0.82, n = 6) — indistinguishable",
        source=None,
    )
    savefig_paper(fig, "issue_650/dv5_selectivity_firing_vs_geometry", dir="figures/")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 5: sycophancy dose trajectory — agreement install vs epoch (plateau).
# ──────────────────────────────────────────────────────────────────────────
def fig_syco_dose() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    chigh = paper_palette_role("primary")
    clow = paper_palette_role("control")
    cband = paper_palette_role("baseline")
    for dose, color in (("high", chigh), ("low", clow)):
        all_curves = []
        for seed in SEEDS:
            p = EVAL / f"syco_dose_trajectory_sycophancy__{dose}__seed{seed}.json"
            d = json.loads(p.read_text())
            recs = sorted(d["epoch_records"], key=lambda r: r["epoch"])
            eps = [r["epoch"] for r in recs]
            delt = [r["delta_agree"] for r in recs]
            all_curves.append((eps, delt))
        # mean curve over seeds (same epoch grid)
        eps0 = all_curves[0][0]
        mat = np.array([c[1] for c in all_curves])
        mean = mat.mean(0)
        sd = mat.std(0, ddof=1)
        ax.plot(eps0, mean, color=color, lw=2.0, marker="o", ms=4, label=f"{dose} dose (n=3 seeds)")
        ax.fill_between(eps0, mean - sd, mean + sd, color=color, alpha=0.18)
    # planned high band lower edge 0.55, low band 0.30-0.45
    ax.axhspan(0.30, 0.45, color=clow, alpha=0.10)
    ax.axhline(0.55, ls="--", lw=1.3, color=cband, label="planned high-dose band floor (0.55)")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("agreement rate over base (Δ)")
    ax.set_ylim(-0.02, 0.62)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    set_title_subtitle(
        ax,
        "Sycophancy install plateaus below the planned high-dose band",
        "rank-1 MLP agreement saturates near +0.37; the high arm never reaches the 0.55 floor",
        source=None,
    )
    savefig_paper(fig, "issue_650/syco_dose_trajectory", dir="figures/")
    plt.close(fig)


if __name__ == "__main__":
    fig_hero()
    fig_dv2()
    fig_dv3_strip()
    fig_dv5()
    fig_syco_dose()
    print("ALL FIGURES WRITTEN")
