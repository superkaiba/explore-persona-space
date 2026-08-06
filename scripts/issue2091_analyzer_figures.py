#!/usr/bin/env python3
"""Clean-result figures for task #2091 (decode-regime comparison).

Six two-panel figures, one per `### <result>` block: each pairs the aggregate
view (left) with the per-context low-level view behind it (right), per the
clean-result spec's low-level-data-plot requirement.

Reads only committed artifacts under ``eval_results/issue_2091/``. Writes to
``figures/issue_2091/`` via ``savefig_paper`` (PNG + PDF + sidecar).

Usage::

    uv run python scripts/issue2091_analyzer_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE matplotlib/numpy: shared-VM thread caps bind in-process (#847)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "eval_results" / "issue_2091"
FIGDIR = "figures/issue_2091"
LAYER = "L19"

SETTING_LABEL = {
    "wildchat": "Everyday prompts",
    "generic": "Everyday prompts",
    "lmsys": "Everyday prompts (banked)",
    "syc_train": "Sycophancy (training)",
    "syc_aita": "Sycophancy (advice forum)",
    "hal_train": "Hallucination (training)",
    "hal_nqopen": "Hallucination (open-domain QA)",
    "hal_simpleqa": "Hallucination (short-fact QA)",
    "evil_train": "Harmful compliance (training)",
    "evil_hhrt": "Harmful compliance (helpful-harmless)",
    "evil_toxicchat": "Harmful compliance (toxic chat)",
}
ORDER = [
    "wildchat",
    "lmsys",
    "syc_train",
    "syc_aita",
    "hal_train",
    "hal_nqopen",
    "hal_simpleqa",
    "evil_train",
    "evil_hhrt",
    "evil_toxicchat",
]
LOWLEVEL = ["wildchat", "syc_train", "hal_simpleqa", "evil_train"]
REGIMES = ("greedy", "avg_k5", "single")
REGIME_LABEL = {
    "greedy": "Greedy (no randomness)",
    "avg_k5": "Five-draw average",
    "single": "Single random draw",
}
FAMILY_LABEL = {
    "pv_projection": "Trait-direction projection",
    "supervised_context": "Supervised prompt read",
    "map_supervised_answer": "Predicted-answer read",
    "oracle_answer": "Observed-answer read",
    "map_pv_projection": "Predicted-answer projection",
}
BEHAVIOR_OF = {"syc": "sycophancy", "hal": "hallucination", "evi": "evil"}


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _regime_colors() -> dict[str, str]:
    pal = paper_palette_blog(3)
    return dict(zip(REGIMES, pal, strict=True))


def _ecdf(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.sort(vals)
    return s, np.arange(1, s.size + 1) / s.size


# ── Figure 1: answer-vector dispersion ────────────────────────────────────────
def fig_dispersion() -> None:
    d = _load("r1_dispersion.json")
    derived = _load("analyzer_derived_reads.json")["truncation_and_length_per_rung"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    keys = [k for k in ORDER if k in d["settings"]]
    med, lo, hi = [], [], []
    for k in keys:
        blk = d["settings"][k][LAYER]
        med.append(blk["summary"]["median"])
        ci = blk["boot_ci_median"]
        lo.append(blk["summary"]["median"] - ci[0])
        hi.append(ci[1] - blk["summary"]["median"])
    y = np.arange(len(keys))
    ax.barh(y, med, xerr=[lo, hi], color=paper_palette_blog(1)[0], height=0.62, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([SETTING_LABEL[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("answer-vector dispersion (mean pairwise cosine distance, 5 draws)")
    ax.set_title("Dispersion spans about fourfold across prompt families", loc="left")

    colors = paper_palette_blog(len(LOWLEVEL))
    for c, k in zip(colors, LOWLEVEL, strict=True):
        vals = np.asarray(d["settings"][k][LAYER]["percontext"]["dispersion"], dtype=float)
        xs, ys = _ecdf(vals)
        mw = derived.get(k, {}).get("median_answer_words")
        lab = f"{SETTING_LABEL[k]} (median {mw:.0f} words)" if mw else SETTING_LABEL[k]
        ax2.plot(xs, ys, color=c, linewidth=1.8, label=lab)
    ax2.set_xlim(0.0, 0.16)
    ax2.set_xlabel("per-prompt answer-vector dispersion")
    ax2.set_ylabel("fraction of prompts at or below")
    ax2.set_title("Per-prompt distributions, not just the medians", loc="left")
    ax2.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    savefig_paper(fig, "hero_dispersion_by_setting", dir=FIGDIR)
    plt.close(fig)


# ── Figure 2: greedy centrality ───────────────────────────────────────────────
def fig_centrality() -> None:
    d = _load("r2_delta.json")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    keys = [k for k in ORDER if k in d["settings"]]
    med, lo, hi, cols = [], [], [], []
    pal = paper_palette_blog(3)
    for k in keys:
        blk = d["settings"][k][LAYER]
        m = blk["median_delta"]
        ci = blk.get("boot_ci_median") or blk.get("delta_boot_ci_median")
        med.append(m)
        lo.append(m - ci[0])
        hi.append(ci[1] - m)
        cols.append(pal[2] if ci[1] < 0 else (pal[1] if ci[0] > 0 else pal[0]))
    y = np.arange(len(keys))
    ax.barh(y, med, xerr=[lo, hi], color=cols, height=0.62, capsize=3)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([SETTING_LABEL[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("greedy-minus-draw closeness (↑ greedy more central)")
    ax.set_title("Greedy is more central in six of eight trait rungs", loc="left")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=pal[1]),
        plt.Rectangle((0, 0), 1, 1, color=pal[2]),
        plt.Rectangle((0, 0), 1, 1, color=pal[0]),
    ]
    ax.legend(
        handles,
        ["interval above zero", "interval below zero", "interval includes zero"],
        loc="lower right",
        fontsize=7.5,
    )

    colors = paper_palette_blog(len(LOWLEVEL))
    for c, k in zip(colors, LOWLEVEL, strict=True):
        vals = np.asarray(d["settings"][k][LAYER]["percontext"]["delta"], dtype=float)
        xs, ys = _ecdf(vals)
        ax2.plot(xs, ys, color=c, linewidth=1.8, label=SETTING_LABEL[k])
    ax2.axvline(0.0, color="#444444", linewidth=1.0)
    ax2.axvline(-0.02, color="#8A8A8A", linewidth=1.0, linestyle="--")
    ax2.set_xlim(-0.08, 0.08)
    ax2.set_xlabel("per-prompt greedy-minus-draw closeness")
    ax2.set_ylabel("fraction of prompts at or below")
    ax2.set_title("Many prompts move the other way", loc="left")
    ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    savefig_paper(fig, "centrality_delta_by_setting", dir=FIGDIR)
    plt.close(fig)


# ── Figure 3: map-quality grid + row/column decomposition ─────────────────────
def fig_map_grid() -> None:
    r4 = _load("r4_grids.json")
    dec = _load("analyzer_derived_reads.json")["r4_row_col_decomposition"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    keys = [k for k in ORDER if k in r4["settings"]] + ["generic"]
    keys = [k for k in keys if k in r4["settings"]]
    colors = _regime_colors()
    width = 0.26
    x = np.arange(len(keys))
    for i, reg in enumerate(REGIMES):
        vals = [r4["settings"][k]["r2_grid"][LAYER][reg][reg] for k in keys]
        ax.bar(x + (i - 1) * width, vals, width=width, color=colors[reg], label=REGIME_LABEL[reg])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SETTING_LABEL[k] for k in keys], rotation=20, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("held-out variance explained (↑ better)")
    ax.set_title("Matched five-draw-average targets predict best", loc="left")
    ax.legend(loc="upper right", ncols=1, fontsize=7.5)

    cg = [dec[k]["eval_column_gain_avg_over_greedy"] for k in keys]
    rg = [dec[k]["fit_row_gain_avg_over_greedy"] for k in keys]
    y = np.arange(len(keys))
    pal = paper_palette_blog(3)
    ax2.barh(y - 0.19, cg, height=0.36, color=pal[1], label="averaging the predicted target")
    ax2.barh(y + 0.19, rg, height=0.36, color=pal[0], label="averaging the target fitted on")
    ax2.set_yticks(y)
    ax2.set_yticklabels([SETTING_LABEL[k] for k in keys], fontsize=7.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("gain in held-out variance explained")
    ax2.set_title("The gain is in the target, not the fitted map", loc="left")
    ax2.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    savefig_paper(fig, "map_quality_grid_and_decomposition", dir=FIGDIR)
    plt.close(fig)


# ── Figure 4: behavioral prediction, raw vs ceiling-normalized ────────────────
def _rho_rows() -> list[tuple[str, str, dict]]:
    r4 = _load("r4_grids.json")
    out = []
    for k, blk in r4["settings"].items():
        if k == "generic":
            continue
        fams = blk[f"behavioral_rho_{LAYER}"]
        out.append((k, BEHAVIOR_OF[k[:3]], fams))
    return out


def fig_behavioral() -> None:
    norm = _load("analyzer_derived_reads.json")["ceiling_normalized_rho"]
    rows = _rho_rows()
    keys = [k for k in ORDER if k in {r[0] for r in rows}]
    fam = "supervised_context"
    colors = _regime_colors()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    width = 0.26
    x = np.arange(len(keys))
    by_key = {k: f for k, _b, f in rows}
    for i, reg in enumerate(REGIMES):
        vals, err_lo, err_hi = [], [], []
        for k in keys:
            cell = by_key[k][fam].get(reg) or {}
            r = cell.get("rho")
            if r is None:
                # not measured for this cell — NaN renders nothing (never a zero bar)
                vals.append(np.nan)
                err_lo.append(0.0)
                err_hi.append(0.0)
                continue
            ci = cell.get("ci95") or [r, r]
            vals.append(r)
            err_lo.append(max(r - ci[0], 0.0))
            err_hi.append(max(ci[1] - r, 0.0))
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            yerr=[err_lo, err_hi],
            capsize=2.5,
            color=colors[reg],
            label=REGIME_LABEL[reg],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SETTING_LABEL[k] for k in keys], rotation=20, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("rank agreement with the judged score (↑ better)")
    ax.set_title("Raw rank agreement with each regime's judged score", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)

    # Right panel: only settings whose judged score has a defined noise ceiling.
    keys2 = [
        k
        for k in keys
        if any(
            (norm.get(f"{k}::{BEHAVIOR_OF[k[:3]]}", {}).get(fam, {}) or {})
            .get(reg, {})
            .get("rho_over_ceiling")
            is not None
            for reg in REGIMES
        )
    ]
    x2 = np.arange(len(keys2))
    for i, reg in enumerate(REGIMES):
        vals = []
        for k in keys2:
            beh = BEHAVIOR_OF[k[:3]]
            cell = (norm.get(f"{k}::{beh}", {}).get(fam, {}) or {}).get(reg) or {}
            v = cell.get("rho_over_ceiling")
            vals.append(v if v is not None else np.nan)
        ax2.bar(
            x2 + (i - 1) * width,
            vals,
            width=width,
            color=colors[reg],
            label=REGIME_LABEL[reg],
        )
    # Flag the two cells whose judged score is floor-censored: HH-RLHF's ceiling
    # is itself near-floor (0.25 greedy / 0.49 averaged), and toxic chat has 70%
    # of prompts at the floor with nine middling prompts.
    NORMALIZED_FLAG = {
        "evil_hhrt": "\n(near-floor ceiling)",
        "evil_toxicchat": "\n(70% of prompts at floor)",
    }
    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [SETTING_LABEL[k] + NORMALIZED_FLAG.get(k, "") for k in keys2],
        rotation=20,
        ha="right",
        rotation_mode="anchor",
    )
    ax2.set_ylabel("share of that score's own noise ceiling")
    ax2.set_title("The same agreement as a share of each score's own ceiling", loc="left")
    fig.tight_layout()
    savefig_paper(fig, "behavior_prediction_raw_and_ceiling_normalized", dir=FIGDIR)
    plt.close(fig)


# ── Figure 5: moderator commonality ───────────────────────────────────────────
def fig_moderators() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    # Cells with at least 100 contexts only; the harmful-compliance transfer
    # rungs carry 9-27 middling contexts and are uninformative (caption states it).
    pairs = [
        ("sycophancy", "wildchat"),
        ("sycophancy", "syc_train"),
        ("sycophancy", "syc_aita"),
        ("hallucination", "wildchat"),
        ("evil", "evil_train"),
    ]
    fams = list(FAMILY_LABEL)
    colors = paper_palette_blog(len(pairs))
    width = 0.15
    x = np.arange(len(fams))
    labels = []
    for i, (beh, setting) in enumerate(pairs):
        d = _load(f"r3_moderators_{beh}.json")
        blk = d["settings"].get(setting)
        if blk is None:
            continue
        c = blk["commonality"]["sigma_a_total"]
        vals = [(c.get(f) or {}).get("r2_full", np.nan) for f in fams]
        beh_lab = "harmful compliance" if beh == "evil" else beh
        lab = (
            f"{SETTING_LABEL[setting]} ({beh_lab})"
            if setting == "wildchat"
            else SETTING_LABEL[setting]
        )
        labels.append(lab)
        ax.bar(x + (i - 2) * width, vals, width=width, color=colors[i], label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [FAMILY_LABEL[f] for f in fams], rotation=20, ha="right", rotation_mode="anchor"
    )
    ax.set_ylabel("share of score variance explained")
    ax.set_title("Both moderators together explain under 5% of score variance", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)

    ys, xs, lo, hi, ticks = [], [], [], [], []
    row = 0
    for beh, setting in pairs:
        d = _load(f"r3_moderators_{beh}.json")
        blk = d["settings"].get(setting)
        if blk is None:
            continue
        c = blk["commonality"]["sigma_a_total"]
        for f in fams:
            comp = (c.get(f) or {}).get("companion_unique_sigma_minus_unique_p")
            if not comp:
                continue
            m = comp["median"]
            ci = comp["ci95"]
            ys.append(row)
            xs.append(m)
            lo.append(max(m - ci[0], 0.0))
            hi.append(max(ci[1] - m, 0.0))
            beh_lab = "harmful compliance" if beh == "evil" else beh
            tick_setting = (
                f"{SETTING_LABEL[setting]} ({beh_lab})"
                if setting == "wildchat"
                else SETTING_LABEL[setting]
            )
            ticks.append(f"{FAMILY_LABEL[f]} — {tick_setting}")
            row += 1
    ax2.errorbar(
        xs,
        ys,
        xerr=[lo, hi],
        fmt="o",
        markersize=4.0,
        markeredgewidth=1.0,
        color=paper_palette_blog(1)[0],
        ecolor="#8A8A8A",
        elinewidth=1.0,
        capsize=2.0,
    )
    ax2.axvline(0.0, color="#444444", linewidth=1.0)
    ax2.set_yticks(list(range(len(ticks))))
    ax2.set_yticklabels(ticks, fontsize=6.2)
    ax2.invert_yaxis()
    ax2.set_xlabel("sampling minus behavioral unique share")
    ax2.set_title("Per-arm contrasts: 19 of 25 intervals cover zero", loc="left")
    fig.tight_layout()
    savefig_paper(fig, "moderator_commonality", dir=FIGDIR)
    plt.close(fig)


# ── Figure 6: polarization ────────────────────────────────────────────────────
def fig_polarization() -> None:
    d = _load("r5_polarization.json")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    panels = [
        p
        for p, blk in d["panels"].items()
        if not blk.get("definitional") and blk.get("g_pol", {}).get("ci95")
    ]
    labels, vals, lo, hi, cols = [], [], [], [], []
    pal = paper_palette_blog(3)
    for p in panels:
        blk = d["panels"][p]
        beh, setting = p.split("::")
        g = blk["g_pol"]
        m = g["value"]
        ci = g["ci95"]
        beh_lab = "harmful compliance" if beh == "evil" else beh
        suffix = " — floor-censored" if blk.get("uninformative") else ""
        labels.append(f"{beh_lab}, {SETTING_LABEL[setting]}{suffix}")
        vals.append(m)
        lo.append(max(m - ci[0], 0.0))
        hi.append(max(ci[1] - m, 0.0))
        cols.append(pal[0] if not blk.get("uninformative") else pal[2])
    y = np.arange(len(labels))
    ax.barh(y, vals, xerr=[lo, hi], color=cols, height=0.6, capsize=3)
    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("middling-draw share above an even split")
    ax.set_title("Middling prompts give middling draws", loc="left")

    pc = d["panels"]["evil::evil_train"]["percontext"]
    mu = np.asarray(pc["mu"], dtype=float)
    sd = np.asarray(pc["sd"], dtype=float)
    ax2.scatter(mu, sd, s=9, alpha=0.35, color=paper_palette_blog(1)[0], linewidths=0.0)
    grid = np.linspace(0, 100, 200)
    ax2.plot(
        grid,
        np.sqrt(grid * (100 - grid)),
        color="#8A8A8A",
        linestyle="--",
        linewidth=1.2,
        label="two-point maximum spread reference",
    )
    ax2.set_xlabel("mean judged score across five draws")
    ax2.set_ylabel("spread of the five draws")
    ax2.set_title("Harmful compliance, training prompts: per-prompt cloud", loc="left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    savefig_paper(fig, "polarization_and_percontext_cloud", dir=FIGDIR)
    plt.close(fig)


def main() -> int:
    set_paper_style("blog")
    fig_dispersion()
    fig_centrality()
    fig_map_grid()
    fig_behavioral()
    fig_moderators()
    fig_polarization()
    print("figures written to", FIGDIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
