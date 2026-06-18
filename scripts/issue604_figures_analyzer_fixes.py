# ruff: noqa: RUF001, RUF003
# Intentional Unicode (Δ, σ, —) in scientific labels.
"""Analyzer-pass figure fixes for task #604 — round 2 regenerates all eight.

Round 1 (commit 0ba61a40c) replaced three figures (seed_stability,
write_match_panel, i474_epoch_ladder). Round 2 folds in the interpretation
critics' figure findings and regenerates the full set:

1.  ``key_match_layer_profile`` — reader-facing panel titles (no issue IDs),
    panels ordered shallow → mid → deep; the epoch-ladder panel's subtitle
    states that its line rides the null band's top edge (the round-1 caption
    claimed "never leaves the band", which is wrong for that panel).
2.  ``seed_stability`` — dots colored by adapter group so the low write
    cluster (EM control + fact line) is identified, not just visible.
3.  ``write_match_panel`` — right panel now plots the LABELED comparator
    (sign-folded |cos| vs each cell's own shared shift direction,
    ``cos_pool_vs_U1_shared_direction``) for BOTH the EM control and the
    saturated endpoint (round 1 plotted ``source_cos`` for the saturated
    bars under the shared-direction y-label); reader-facing tick labels.
4.  ``dose_rotation_scatter`` — hollow joint-cell markers get explicit
    ``linewidths`` (the blog style zeroes scatter edge widths, so they were
    invisible — task #536 pitfall); legend now explains filled vs hollow;
    shorter y-label so nothing clips.
5.  ``i474_epoch_ladder`` — shorter y-label (was clipped at the left edge);
    no ``tight_layout`` fight with constrained_layout.
6.  ``spectral_concentration`` — reader-facing legend labels.
7.  ``constancy_histogram`` — reader-facing legend labels; subtitle notes
    the hot-lr fact line is not in this read.
8.  ``selectivity_margin_bars`` — reader-facing tick labels, shorter y-label.

Reads eval_results/issue_604/{key_match,write_match,rotation,
functional_constancy,selectivity}.json + spectra/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "eval_results" / "issue_604"
FIG = PROJECT_ROOT / "figures" / "issue_604"

# Reader-facing line labels — no project-internal issue IDs (critic round 1).
LINE_LABELS = {
    "dial527": "dose dial — shallow window",
    "dial550": "dose dial — mid window",
    "dial538": "dose dial — deep window",
    "i474": "epoch ladder",
    "i518": "cross-behavior",
    "i519": "saturated marker endpoint",
    "i521": "EM control (no persona)",
    "i541": "fact line (hot lr)",
}
PROFILE_ORDER = ["dial527", "dial550", "dial538", "i474", "i518", "i519"]
PAIR_LABELS = {
    "florist__medical_doctor": "florist / medical doctor pair",
    "librarian__police_officer": "librarian / police officer pair",
}
# Seed-stability grouping: which adapter family a (line, group) belongs to.
SEED_GROUPS = [
    ("marker dose dial", ("dial527", "dial538", "dial550"), "primary"),
    ("saturated marker endpoint", ("i519",), "accent"),
    ("EM control (no persona)", ("i521",), "control"),
    ("fact line (hot lr)", ("i541",), "baseline"),
]


def _group_of(line: str) -> tuple[str, str]:
    """Return (group label, palette role) for a seed-stability line."""
    for label, lines, role in SEED_GROUPS:
        if line in lines:
            return label, role
    return line, "neutral"


def fig_key_profile(km: dict) -> None:
    """Per-line key-match layer profile with the wrong-context null band."""
    by_line: dict[str, list[dict]] = defaultdict(list)
    for cell in km["cells"]:
        for src in cell["per_source"]:
            attn = src.get("stacks", {}).get("attn_key", {}).get("attn")
            if attn:
                by_line[cell["line"]].append(attn)
    lines = [ln for ln in PROFILE_ORDER if ln in by_line]
    fig, axes = plt.subplots(1, len(lines), figsize=(4.0 * len(lines), 3.6), squeeze=False)
    for ax, line in zip(axes[0], lines, strict=True):
        rows = by_line[line]
        layers = sorted({r["layer"] for entry in rows for r in entry["layers"]})
        src_mean, null_lo, null_hi = [], [], []
        for layer in layers:
            vals = [r for entry in rows for r in entry["layers"] if r["layer"] == layer]
            src_mean.append(np.mean([v["cos_src_abs"] for v in vals]))
            null_lo.append(np.mean([v["null_p50"] for v in vals]))
            null_hi.append(np.mean([v["null_p95"] for v in vals]))
        ax.fill_between(
            layers,
            null_lo,
            null_hi,
            color=paper_palette_role("neutral"),
            alpha=0.35,
            label="wrong-context null (p50–p95)",
        )
        ax.plot(
            layers, src_mean, color=paper_palette_role("primary"), lw=1.8, label="source context"
        )
        ax.set_xlabel("layer")
        ax.set_ylabel("|cos(top key, context)|")
        sub = "module-input space (mean over cells)"
        if line == "i474":
            sub = "module-input space — rides the band's top edge"
        set_title_subtitle(ax, LINE_LABELS.get(line, line), sub)
        ax.legend(fontsize=7)
    savefig_paper(fig, "key_match_layer_profile", dir=FIG)
    plt.close(fig)


def fig_seed_stability(sel: dict) -> None:
    """Cross-seed key vs write |cos|, dots colored by adapter group."""
    rng = np.random.default_rng(42)
    pts: list[tuple[str, str, float, float]] = []  # (group, role, key, write)
    for g in sel["seed_stability"]:
        label, role = _group_of(g["line"])
        for p in g["pairs"]:
            if p.get("key_abs_cos_band_mean") is None:
                continue
            pts.append((label, role, p["key_abs_cos_band_mean"], p["write_abs_cos_band_mean"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    seen: set[str] = set()
    for x0, idx in ((0.0, 2), (1.0, 3)):
        vals = [(p[0], p[1], p[idx]) for p in pts]
        for label, role, v in vals:
            ax.scatter(
                x0 + rng.uniform(-0.12, 0.12),
                v,
                s=16,
                alpha=0.7,
                color=paper_palette_role(role),
                label=label if (label not in seen and x0 == 1.0) else None,
            )
            if x0 == 1.0:
                seen.add(label)
        med = float(np.median([v for *_, v in vals]))
        ax.plot([x0 - 0.22, x0 + 0.22], [med, med], color="black", lw=1.4)
        ax.annotate(f"median {med:.3f}", (x0 + 0.26, med), fontsize=8, va="center")
    ax.set_xticks([0.0, 1.0], ["key\n(top input direction)", "write\n(top output direction)"])
    ax.set_xlim(-0.5, 1.95)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("cross-seed |cos|, layer-band mean")
    ax.legend(fontsize=7, loc="center right")
    set_title_subtitle(
        ax,
        "Same data, different seed: the write reproduces only where the update concentrates",
        "one dot per seed pair within a training group (n = 69 each side); "
        "low write cluster = diffuse-update lines",
    )
    savefig_paper(fig, "seed_stability", dir=FIG)
    plt.close(fig)


def fig_write_match(wm: dict) -> None:
    """Dial scatter vs dose + control bars on the labeled shared-direction comparator."""
    dial, controls = [], []
    for cell in wm["cells"]:
        if "per_source" in cell:
            for rec in cell["per_source"]:
                dial.append(
                    (
                        cell["dose"].get(rec["source"]),
                        rec["cos_abs"],
                        rec["null_p5"],
                        rec["null_p95"],
                    )
                )
        elif "variants" in cell:
            v = cell["variants"].get("same")
            if not isinstance(v, dict):
                continue
            comp = v.get("cos_pool_vs_U1_shared_direction")
            if comp is None:
                continue
            grp = "EM control" if cell["line"] == "i521" else "saturated"
            seed = cell["cell_id"].rsplit("seed", 1)[-1]
            controls.append((grp, seed, abs(comp)))
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 4.0), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    for dose, _cos_abs, p5, p95 in dial:
        ax.plot([dose, dose], [p5, p95], color=paper_palette_role("neutral"), lw=1.0, alpha=0.5)
    ax.scatter(
        [d for d, *_ in dial],
        [c for _, c, *_ in dial],
        s=16,
        color=paper_palette_role("primary"),
        zorder=3,
        label="weight-space write vs source's measured shift",
    )
    ax.set_xlabel("realized implant depth (nat, re-measured per cell)")
    ax.set_ylabel("|cos(pooled write, measured shift)|")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left", fontsize=8)
    set_title_subtitle(
        ax,
        "The write does not match the measured shift",
        "grey spans = wrong-context null p5–p95 within each cell (n = 72 reads)",
    )
    order = [c for c in controls if c[0] == "EM control"] + [
        c for c in controls if c[0] != "EM control"
    ]
    xs = np.arange(len(order))
    em_color = paper_palette_role("control")
    sat_color = paper_palette_role("baseline")
    colors = [em_color if grp == "EM control" else sat_color for grp, _, _ in order]
    ax2.bar(xs, [v for _, _, v in order], color=colors)
    ax2.axhline(0.5, color="black", lw=1.0, ls="--")
    ax2.annotate("positive-control bar (0.5)", (0.02, 0.51), fontsize=7)
    ax2.set_xticks(xs, [seed for _, seed, _ in order], fontsize=8)
    ax2.set_xlabel("training seed")
    handles = [
        Line2D([], [], marker="s", ls="none", mfc=em_color, mec=em_color, label="EM control"),
        Line2D(
            [],
            [],
            marker="s",
            ls="none",
            mfc=sat_color,
            mec=sat_color,
            label="saturated marker endpoint",
        ),
    ]
    ax2.legend(handles=handles, fontsize=7, loc="center right")
    ax2.set_ylim(0, 0.6)
    ax2.set_ylabel("|cos(pooled write, shared direction)|")
    set_title_subtitle(
        ax2,
        "Positive control fails",
        "sign-folded |cos| vs own shared direction, layer 14",
    )
    savefig_paper(fig, "write_match_panel", dir=FIG)
    plt.close(fig)


def fig_dose_rotation(rot: dict) -> None:
    """Δcos vs realized implant depth; hollow joint markers now visible."""
    prim = rot["primary_30_clean_single_source"]["cells"]
    joint = rot["secondary_joint_per_source"]["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = {
        "florist__medical_doctor": paper_palette_role("primary"),
        "librarian__police_officer": paper_palette_role("accent"),
    }
    for rows, hollow in ((prim, False), (joint, True)):
        for r in rows:
            c = colors.get(r["pair"], paper_palette_role("neutral"))
            ax.scatter(
                r["dose_delta_logp_marker"],
                r["delta_cos"],
                facecolors="none" if hollow else c,
                edgecolors=c,
                linewidths=1.3,  # blog style zeroes edge widths — keep hollow visible
                s=46,
                zorder=3,
            )
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8, zorder=1)
    ax.set_xlabel("realized implant depth (nat, re-measured per cell)")
    ax.set_ylabel("Δ|cos| (contrast − raw)")
    trend = rot["primary_30_clean_single_source"]["trend"]
    rho = trend.get("spearman_rho")
    sub = "rotation of the key toward source-minus-negatives"
    if rho is not None:
        sub += f"; Spearman {rho:+.2f} on the filled cells"
    set_title_subtitle(ax, "Key rotation grows with implant depth?", sub)
    handles = [
        Line2D([], [], marker="o", ls="none", mfc=c, mec=c, label=PAIR_LABELS.get(pair, pair))
        for pair, c in colors.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc="#444444",
            mec="#444444",
            label="single-source cell (filled, n=30)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            mfc="none",
            mec="#444444",
            mew=1.3,
            label="joint cell, per-source read (hollow)",
        ),
    ]
    ax.legend(handles=handles, fontsize=7)

    ax2 = axes[1]
    for key, role, label in (
        ("cos_contrast", "primary", "vs source-minus-negatives"),
        ("cos_raw", "baseline", "vs raw source context"),
        ("cos_placebo", "control", "vs placebo contrast"),
    ):
        xs = [r["dose_delta_logp_marker"] for r in prim]
        ys = [r[key] for r in prim]
        ax2.scatter(xs, ys, color=paper_palette_role(role), s=30, label=label, alpha=0.85)
    ax2.set_xlabel("realized implant depth (nat)")
    ax2.set_ylabel("|cos(key, direction)| (band mean)")
    set_title_subtitle(
        ax2, "Component cosines", "true rotation = contrast term rises above placebo"
    )
    ax2.legend()
    savefig_paper(fig, "dose_rotation_scatter", dir=FIG)
    plt.close(fig)


def fig_i474_ladder(rot: dict) -> None:
    """Contrastive vs positives-only Δcos by epoch, un-clipped y-label."""
    lad = rot["i474_epoch_ladder"]
    reads = lad["reads"]
    agg = lad["aggregate"]
    by: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for r in reads:
        by.setdefault((r["arm"], r["source"]), []).append((r["epoch"], r["delta_cos"]))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    seen: set[str] = set()
    for (arm, _src), p in sorted(by.items()):
        p.sort()
        color = paper_palette_role("primary" if arm == "loc" else "accent")
        label = (
            ("contrastive arm" if arm == "loc" else "positives-only arm")
            if arm not in seen
            else None
        )
        seen.add(arm)
        ax.plot([e for e, _ in p], [d for _, d in p], color=color, alpha=0.55, lw=1.0, label=label)
    ax.set_xticks([1, 2, 3, 5])
    ax.set_xlabel("training epochs")
    ax.set_ylabel("Δ|cos| toward matched contrast")
    ax.legend(fontsize=8)
    set_title_subtitle(
        ax,
        "Does contrastive training rotate the key with epochs?",
        f"one line per source; paired slope difference {agg['mean']:+.4f}/epoch, "
        f"CI [{agg['ci_lo_mean']:+.4f}, {agg['ci_hi_mean']:+.4f}]",
    )
    savefig_paper(fig, "i474_epoch_ladder", dir=FIG)
    plt.close(fig)


def fig_spectra(rot: dict | None) -> None:
    """Spectral concentration by layer + effective rank vs dose, plain labels."""
    spectra_files = sorted((OUT / "spectra").glob("*/*.json"))
    per_line: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for path in spectra_files:
        payload = json.loads(path.read_text())
        line = payload["cell"]["line"]
        for rec in payload["layers"]:
            for st in rec["stacks"]:
                if st["stack"] == "attn_key":
                    per_line[line].append((rec["layer"], st["top1_energy"], st["effective_rank"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    order = [ln for ln in LINE_LABELS if ln in per_line]
    for line in order:
        vals = per_line[line]
        layers = sorted({v[0] for v in vals})
        m = [np.mean([v[1] for v in vals if v[0] == layer]) for layer in layers]
        axes[0].plot(layers, m, label=LINE_LABELS.get(line, line), lw=1.5)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("top-1 energy of stacked attention update")
    set_title_subtitle(axes[0], "Spectral concentration by layer", "stacked q/k/v update")
    axes[0].legend(fontsize=7)
    ax2 = axes[1]
    if rot:
        prim = rot["primary_30_clean_single_source"]["cells"]
        doses = {(r["line"], r["cell_id"]): r["dose_delta_logp_marker"] for r in prim}
        xs, ys = [], []
        for path in spectra_files:
            payload = json.loads(path.read_text())
            key = (payload["cell"]["line"], payload["cell"]["cell_id"])
            if key not in doses:
                continue
            effs = [
                st["effective_rank"]
                for rec in payload["layers"]
                for st in rec["stacks"]
                if st["stack"] == "attn_key" and rec["layer"] in range(14, 25)
            ]
            if effs:
                xs.append(doses[key])
                ys.append(float(np.mean(effs)))
        ax2.scatter(xs, ys, color=paper_palette_role("primary"), s=34)
        ax2.set_xlabel("realized implant depth (nat)")
        ax2.set_ylabel("effective rank (band mean)")
        set_title_subtitle(ax2, "Effective rank vs dose", "stacked attention update, L14–L24")
    savefig_paper(fig, "spectral_concentration", dir=FIG)
    plt.close(fig)


def fig_constancy(fc: dict) -> None:
    """Wang-et-al.-style constancy histogram, plain-English legend."""
    by_line = defaultdict(list)
    for cell in fc["cells"]:
        by_line[cell["line"]].append(cell["band_mean_pairwise_abs_cos"])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for line in [ln for ln in LINE_LABELS if ln in by_line]:
        vals = by_line[line]
        ax.hist(vals, bins=np.linspace(0, 1, 21), alpha=0.55, label=LINE_LABELS.get(line, line))
    ax.set_xlabel("pairwise |cos| of adapter output across contexts (band mean)")
    ax.set_ylabel("cells")
    set_title_subtitle(
        ax,
        "Is the adapter's output a constant direction?",
        "1.0 = constant steering vector; hot-lr fact line not read (sources outside the bank)",
    )
    ax.legend(fontsize=7)
    savefig_paper(fig, "constancy_histogram", dir=FIG)
    plt.close(fig)


def fig_selectivity(km: dict) -> None:
    """Selectivity margin per line, reader-facing tick labels."""
    by_line = defaultdict(list)
    for cell in km["cells"]:
        for src in cell["per_source"]:
            attn = src.get("stacks", {}).get("attn_key", {}).get("attn")
            if not attn:
                continue
            band = [r for r in attn["layers"] if r["layer"] in km["layer_band"]]
            if band:
                by_line[cell["line"]].append(
                    float(np.mean([r["selectivity_margin"] for r in band]))
                )
    lines = [ln for ln in PROFILE_ORDER if ln in by_line]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    xs = np.arange(len(lines))
    means = [float(np.mean(by_line[ln])) for ln in lines]
    err = [
        max(0.0, 1.96 * float(np.std(by_line[ln], ddof=1)) / np.sqrt(len(by_line[ln])))
        if len(by_line[ln]) > 1
        else 0.0
        for ln in lines
    ]
    ax.bar(xs, means, yerr=err, color=paper_palette_role("primary"), capsize=3)
    for x, ln in zip(xs, lines, strict=True):
        ax.scatter(
            np.full(len(by_line[ln]), x) + np.linspace(-0.15, 0.15, len(by_line[ln])),
            by_line[ln],
            color=paper_palette_role("neutral"),
            s=12,
            zorder=3,
        )
    ax.axhline(0.0, color=paper_palette_role("neutral"), lw=0.8)
    ax.set_xticks(xs, [LINE_LABELS.get(ln, ln) for ln in lines], rotation=20, ha="right")
    ax.set_ylabel("selectivity margin (|cos|)")
    set_title_subtitle(
        ax,
        "Does the key single out its own source?",
        "source − best non-source |cos|, band mean L14–L24, one dot per cell",
    )
    savefig_paper(fig, "selectivity_margin_bars", dir=FIG)
    plt.close(fig)


def main() -> None:
    """Regenerate all eight #604 figures with the round-2 critic fixes."""
    set_paper_style("blog")
    km = json.loads((OUT / "key_match.json").read_text())
    sel = json.loads((OUT / "selectivity.json").read_text())
    wm = json.loads((OUT / "write_match.json").read_text())
    rot = json.loads((OUT / "rotation.json").read_text())
    fc = json.loads((OUT / "functional_constancy.json").read_text())
    fig_key_profile(km)
    fig_seed_stability(sel)
    fig_write_match(wm)
    fig_dose_rotation(rot)
    fig_i474_ladder(rot)
    fig_spectra(rot)
    fig_constancy(fc)
    fig_selectivity(km)
    print("done:", FIG)


if __name__ == "__main__":
    main()
