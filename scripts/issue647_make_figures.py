#!/usr/bin/env python3
"""task #647 clean-result figure generation.

Reads ``eval_results/issue_647/nonstylized_predictor_sweep.json`` (the full
192-row grid) and ``eval_results/issue_522/js_regression.json`` (the JS
comparison anchor), and produces four figures under ``figures/issue_647/``:

  side_by_side_cv_r2_headline.png       — HERO. Per-metric CV R² bars, full
      vs nonstylized panel at L22/loc-ep1, with panel-row bootstrap 95% CIs,
      and #522's JS bars (full / nonstylized) overlaid as the comparison anchor.
  cv_r2_full_vs_nonsty_by_epoch.png     — 4-panel grid (one per epoch in
      {1,2,3,5}) of the same side-by-side, headline layer L22 only.
  nonsty_cv_r2_heatmap_by_layer_epoch.png — 4 heatmaps (one per metric),
      L19-L24 × epoch {1,2,3,5}, CV R² on the nonstylized panel.
  headline_pair_scatter_nonsty.png      — per-pair scatter of gauss_kl distance
      (L22) vs ΔG (loc-ep1), nonstylized 156 pairs only, length-partialled
      regression line.

Run from repo root or worktree:
    uv run python scripts/issue647_make_figures.py

Plain-English metric / persona names everywhere (clean-result-critic Lens 3 /
interpretation-critic Lens 6); no opaque codes in user-facing labels.
"""

# ruff: noqa: RUF001, RUF002, RUF003 (research notation: ρ, Δ in strings)

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

WT = Path(__file__).resolve().parent.parent
EVAL_647 = WT / "eval_results" / "issue_647"
G_EP1 = WT / "eval_results" / "issue_474" / "cross_eval" / "loc_ep1" / "G_logprob_matrix.json"
D_MATRIX = WT / "eval_results" / "issue_406" / "divergence" / "D_matrix.json"

STY_CIDS = frozenset({"A3", "A4", "A5"})  # A3 pirate / A4 comedian / A5 villain

# Plain-English metric names (clean-result-critic Lens 3).
METRIC_LABEL = {
    "gauss_kl": "Gaussian KL",
    "mmd": "MMD",
    "wass2": "2-Wasserstein",
    "cosine": "Cosine",
}
METRICS = ("gauss_kl", "mmd", "wass2", "cosine")
EPOCHS = (1, 2, 3, 5)
LAYERS = (19, 20, 21, 22, 23, 24)
HEADLINE_LAYER = 22
HEADLINE_EPOCH = 1

# #522 JS bars (loc ep1) — the comparison anchor (plan §5, hero overlay).
JS_FULL = {"cv_r2": 0.2411, "lo": 0.121, "hi": 0.338}
JS_NONSTY = {"cv_r2": -0.0218, "lo": -0.151, "hi": 0.078}


# ───────────────────────── loaders ─────────────────────────


def _load_sweep() -> dict:
    p = EVAL_647 / "nonstylized_predictor_sweep.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}; run issue647_nonstylized_predictor_sweep.py --mode full first."
        )
    return json.loads(p.read_text())


def _row(rows: list[dict], *, metric: str, layer: int, panel: str, epoch: int) -> dict:
    """Return the single sweep row for (metric, layer, panel, epoch); raise if
    absent or not status=ok."""
    matches = [
        r
        for r in rows
        if r.get("metric") == metric
        and r.get("layer") == layer
        and r.get("panel") == panel
        and r.get("epoch") == epoch
    ]
    if not matches:
        raise RuntimeError(f"No row for metric={metric} L{layer} {panel} ep{epoch}")
    r = matches[0]
    if r.get("status") != "ok":
        raise RuntimeError(f"Row metric={metric} L{layer} {panel} ep{epoch} status={r['status']}")
    return r


def _pairs(cond_ids: list[str], *, nonstylized: bool) -> list[tuple[str, str]]:
    out = []
    for a in cond_ids:
        for b in cond_ids:
            if a == b:
                continue
            if nonstylized and (a in STY_CIDS or b in STY_CIDS):
                continue
            out.append((a, b))
    return out


# ───────────────────────── figure 1: hero side-by-side ─────────────────────────


def side_by_side_cv_r2_headline() -> None:
    """HERO: per-metric CV R² — full panel vs nonstylized panel at L22/ep1 —
    with panel-row bootstrap 95% CI error bars on both, plus the #522 JS bars
    (full / nonstylized) overlaid as the comparison anchor. Dashed line at 0.

    Read: if the nonstylized activation bars sit near the JS-nonstylized bar
    (≈0), the predictors collapsed like JS (H_stylized-only); if they stay well
    above, activation geometry retains within-nonstylized signal (H_geometry-real).
    """
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    rows = _load_sweep()["rows"]

    # x positions: 4 activation metrics + JS anchor.
    groups = [*METRICS, "js"]
    group_labels = [*[METRIC_LABEL[m] for m in METRICS], "Full-response JS\n(output distribution)"]
    x = np.arange(len(groups))
    w = 0.38

    full_cv, full_err = [], [[], []]
    nons_cv, nons_err = [], [[], []]
    for m in METRICS:
        rf = _row(rows, metric=m, layer=HEADLINE_LAYER, panel="full", epoch=HEADLINE_EPOCH)
        rn = _row(rows, metric=m, layer=HEADLINE_LAYER, panel="nonstylized", epoch=HEADLINE_EPOCH)
        cf = rf["point_estimate"]["cv_r2"]
        cn = rn["point_estimate"]["cv_r2"]
        full_cv.append(cf)
        nons_cv.append(cn)
        full_err[0].append(cf - rf["panel_ci"]["cv_r2"]["lo"])
        full_err[1].append(rf["panel_ci"]["cv_r2"]["hi"] - cf)
        nons_err[0].append(cn - rn["panel_ci"]["cv_r2"]["lo"])
        nons_err[1].append(rn["panel_ci"]["cv_r2"]["hi"] - cn)
    # JS anchor bars.
    full_cv.append(JS_FULL["cv_r2"])
    nons_cv.append(JS_NONSTY["cv_r2"])
    full_err[0].append(JS_FULL["cv_r2"] - JS_FULL["lo"])
    full_err[1].append(JS_FULL["hi"] - JS_FULL["cv_r2"])
    nons_err[0].append(JS_NONSTY["cv_r2"] - JS_NONSTY["lo"])
    nons_err[1].append(JS_NONSTY["hi"] - JS_NONSTY["cv_r2"])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    c_full = paper_palette_role("primary")
    c_nons = paper_palette_role("baseline")

    ax.bar(
        x - w / 2,
        full_cv,
        width=w,
        color=c_full,
        edgecolor="white",
        linewidth=0.7,
        label="Full panel (16 personas, 240 pairs)",
    )
    ax.errorbar(
        x - w / 2,
        full_cv,
        yerr=np.array(full_err),
        fmt="none",
        ecolor="#2a2a2a",
        elinewidth=0.9,
        capsize=2.5,
    )
    ax.bar(
        x + w / 2,
        nons_cv,
        width=w,
        color=c_nons,
        edgecolor="white",
        linewidth=0.7,
        label="Nonstylized panel (13 personas, 156 pairs)",
    )
    ax.errorbar(
        x + w / 2,
        nons_cv,
        yerr=np.array(nons_err),
        fmt="none",
        ecolor="#2a2a2a",
        elinewidth=0.9,
        capsize=2.5,
    )

    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel("LOCO CV R² predicting marker-leakage transfer ΔG")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    set_title_subtitle(
        ax,
        "Activation-predictor CV R² on the full vs nonstylized panel, against the JS anchor",
        "Full vs nonstylized panel · layer 22 · loc-arm epoch 1 · "
        "panel-row bootstrap 95% CI (n_boot=2000, seed=42)",
        source="activation metrics: cached #502 residual streams · "
        "JS anchor: #522 base-model output-distribution",
    )
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.set_constrained_layout(False)
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.09, right=0.97)

    savefig_paper(fig, "issue_647/side_by_side_cv_r2_headline")
    plt.close(fig)


# ───────────────────── figure 2: by-epoch side-by-side grid ─────────────────────


def cv_r2_full_vs_nonsty_by_epoch() -> None:
    """4-panel grid (one per epoch in {1,2,3,5}) of the headline-layer (L22)
    full-vs-nonstylized side-by-side CV R² bars with panel-row 95% CIs."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    rows = _load_sweep()["rows"]

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.2), sharey=True)
    x = np.arange(len(METRICS))
    w = 0.38
    c_full = paper_palette_role("primary")
    c_nons = paper_palette_role("baseline")

    for ax, ep in zip(axes, EPOCHS, strict=True):
        full_cv, full_err = [], [[], []]
        nons_cv, nons_err = [], [[], []]
        for m in METRICS:
            rf = _row(rows, metric=m, layer=HEADLINE_LAYER, panel="full", epoch=ep)
            rn = _row(rows, metric=m, layer=HEADLINE_LAYER, panel="nonstylized", epoch=ep)
            cf = rf["point_estimate"]["cv_r2"]
            cn = rn["point_estimate"]["cv_r2"]
            full_cv.append(cf)
            nons_cv.append(cn)
            full_err[0].append(cf - rf["panel_ci"]["cv_r2"]["lo"])
            full_err[1].append(rf["panel_ci"]["cv_r2"]["hi"] - cf)
            nons_err[0].append(cn - rn["panel_ci"]["cv_r2"]["lo"])
            nons_err[1].append(rn["panel_ci"]["cv_r2"]["hi"] - cn)

        ax.bar(x - w / 2, full_cv, width=w, color=c_full, edgecolor="white", linewidth=0.6)
        ax.errorbar(
            x - w / 2,
            full_cv,
            yerr=np.array(full_err),
            fmt="none",
            ecolor="#2a2a2a",
            elinewidth=0.8,
            capsize=2.0,
        )
        ax.bar(x + w / 2, nons_cv, width=w, color=c_nons, edgecolor="white", linewidth=0.6)
        ax.errorbar(
            x + w / 2,
            nons_cv,
            yerr=np.array(nons_err),
            fmt="none",
            ecolor="#2a2a2a",
            elinewidth=0.8,
            capsize=2.0,
        )
        ax.axhline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=8, rotation=20, ha="right")
        ax.set_title(f"loc-arm epoch {ep}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("LOCO CV R² vs ΔG")
    # One shared legend.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c_full),
        plt.Rectangle((0, 0), 1, 1, color=c_nons),
    ]
    axes[-1].legend(
        handles,
        ["Full panel (240 pairs)", "Nonstylized panel (156 pairs)"],
        loc="upper right",
        frameon=False,
        fontsize=8,
    )

    fig.suptitle(
        "The full-vs-nonstylized CV R² gap holds at every loc-arm training amount (layer 22)",
        x=0.04,
        y=1.00,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.text(
        0.04,
        0.95,
        "Panel-row bootstrap 95% CIs · n_boot = 2000 · seed = 42",
        ha="left",
        fontsize=9.5,
        color="#454545",
    )
    fig.set_constrained_layout(False)
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.05, right=0.99, wspace=0.10)

    savefig_paper(fig, "issue_647/cv_r2_full_vs_nonsty_by_epoch")
    plt.close(fig)


# ───────────────────── figure 3: nonstylized L×epoch heatmaps ─────────────────────


def nonsty_cv_r2_heatmap_by_layer_epoch() -> None:
    """4 heatmaps (one per metric), L19-L24 × epoch {1,2,3,5}, nonstylized-panel
    CV R². Surfaces whether any within-nonstylized signal is layer/epoch-localized
    or uniformly absent."""
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    rows = _load_sweep()["rows"]

    # Build a (n_layers × n_epochs) grid per metric; shared color scale.
    grids: dict[str, np.ndarray] = {}
    for m in METRICS:
        grid = np.full((len(LAYERS), len(EPOCHS)), np.nan)
        for i, L in enumerate(LAYERS):
            for j, ep in enumerate(EPOCHS):
                r = _row(rows, metric=m, layer=L, panel="nonstylized", epoch=ep)
                grid[i, j] = r["point_estimate"]["cv_r2"]
        grids[m] = grid
    allvals = np.concatenate([g.ravel() for g in grids.values()])
    vmax = float(np.nanmax(np.abs(allvals)))
    vmin = -vmax  # symmetric diverging scale centered at 0

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.8))
    im = None
    for ax, m in zip(axes, METRICS, strict=True):
        grid = grids[m]
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(EPOCHS)))
        ax.set_xticklabels([f"ep {ep}" for ep in EPOCHS], fontsize=8)
        ax.set_yticks(np.arange(len(LAYERS)))
        ax.set_yticklabels([f"L{L}" for L in LAYERS], fontsize=8)
        ax.set_title(METRIC_LABEL[m], fontsize=10)
        ax.set_xlabel("loc-arm training amount", fontsize=8.5)
        for i in range(len(LAYERS)):
            for j in range(len(EPOCHS)):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color="white" if abs(v) > 0.55 * vmax else "#1a1a1a",
                    )
    axes[0].set_ylabel("Extraction layer")
    cbar = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.012)
    cbar.set_label("Nonstylized-panel LOCO CV R² vs ΔG", fontsize=9)

    fig.suptitle(
        "Within-nonstylized predictive skill across layers L19-L24 and loc-arm training amount",
        x=0.04,
        y=1.02,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )

    savefig_paper(fig, "issue_647/nonsty_cv_r2_heatmap_by_layer_epoch")
    plt.close(fig)


# ───────────────────── figure 4: headline-pair scatter ─────────────────────


def _raw_ols_line(xv: np.ndarray, yv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, y_fit) for a plain raw-space OLS line of yv on xv.

    Drawn faithfully in the SAME raw axes as the scatter — the trend the
    plotted points actually show. The reported Spearman ρ / LOCO CV R² in the
    caption are the length-partialled rank-space statistics (the analysis DV);
    the caption labels the line as the raw trend so the two are not conflated.
    """
    finite = np.isfinite(xv) & np.isfinite(yv)
    b, a = np.polyfit(xv[finite], yv[finite], 1)
    x_grid = np.linspace(float(xv[finite].min()), float(xv[finite].max()), 100)
    return x_grid, a + b * x_grid


def headline_pair_scatter_nonsty() -> None:
    """Per-pair scatter: gauss_kl distance (L22) vs ΔG (loc-ep1), nonstylized
    156 pairs only, with a raw-space OLS trend line — the raw companion to the
    headline CV R² number (show raw alongside processed). The caption carries
    the length-partialled ρ / LOCO CV R² (the analysis DV)."""
    set_paper_style("blog")
    sweep = _load_sweep()
    rows = sweep["rows"]
    # Per-pair distances: recompute the L22 gauss_kl metric matrix on the cached
    # residuals (same path the sweep used), restricted to nonstylized pairs.
    sys.path.insert(0, str(WT / "scripts"))
    import issue493_extraction_metric_bakeoff as bakeoff
    import issue511_probe_count_sweep as sweep_mod

    act, cond_ids = sweep_mod.load_activations_slice("last_prompt", HEADLINE_LAYER)
    payload = bakeoff._compute_metric_matrix(
        activations=act,
        cond_ids=cond_ids,
        metric="gauss_kl",
        extraction_point="last_prompt",
        pca_k=bakeoff.PCA_DEFAULT_K,
        variant="raw",
    )
    pairs = _pairs(list(cond_ids), nonstylized=True)
    xv = bakeoff._materialize_predictor_vector(payload, pairs, sub_predictor=None)
    G = json.loads(G_EP1.read_text())["G"]
    prompt_tokens = json.loads(D_MATRIX.read_text())["prompt_tokens"]
    yv = np.array([G[a][b]["delta_g"] for a, b in pairs], dtype=np.float64)
    # prompt_tokens loaded for parity with the regression substrate; the raw
    # scatter line is a plain OLS trend (the length partial lives in the
    # rank-space ρ / CV R² reported in the caption, not in the drawn line).
    _ = prompt_tokens

    r = _row(
        rows, metric="gauss_kl", layer=HEADLINE_LAYER, panel="nonstylized", epoch=HEADLINE_EPOCH
    )
    rho = r["point_estimate"]["rho"]
    p = r["point_estimate"]["p"]
    cv = r["point_estimate"]["cv_r2"]
    cv_lo = r["panel_ci"]["cv_r2"]["lo"]
    cv_hi = r["panel_ci"]["cv_r2"]["hi"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.scatter(
        xv,
        yv,
        s=24,
        alpha=0.7,
        color=paper_palette_role("baseline"),
        edgecolors="white",
        linewidths=0.5,
        label=f"nonstylized pairs (n={len(pairs)})",
    )
    x_line, y_line = _raw_ols_line(xv, yv)
    ax.plot(
        x_line,
        y_line,
        color=paper_palette_role("accent"),
        linewidth=1.6,
        label="raw OLS trend",
    )
    ax.set_xlabel("Gaussian KL activation distance (layer 22, cloud-aware ridge)")
    ax.set_ylabel("Marker-leakage transfer ΔG (nats; from #474)")

    set_title_subtitle(
        ax,
        "Gaussian KL activation distance vs marker-leakage transfer, normal personas only",
        f"Spearman ρ = {rho:.2f} (length-partialled) · LOCO CV R² = {cv:.2f} "
        f"[{cv_lo:.2f}, {cv_hi:.2f}] · n = {len(pairs)}, p = {p:.1e}",
        source="nonstylized 156-pair panel · 13×12 personas · loc-arm epoch 1",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    savefig_paper(fig, "issue_647/headline_pair_scatter_nonsty")
    plt.close(fig)


def main() -> int:
    side_by_side_cv_r2_headline()
    cv_r2_full_vs_nonsty_by_epoch()
    nonsty_cv_r2_heatmap_by_layer_epoch()
    headline_pair_scatter_nonsty()
    print(
        "Wrote figures/issue_647/{side_by_side_cv_r2_headline, "
        "cv_r2_full_vs_nonsty_by_epoch, nonsty_cv_r2_heatmap_by_layer_epoch, "
        "headline_pair_scatter_nonsty}.{png,pdf,meta.json}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
