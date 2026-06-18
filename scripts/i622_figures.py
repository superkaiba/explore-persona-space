#!/usr/bin/env python3
# em-dash + Qwen marker token " ※" intentional
"""Task #622 — off-pod figures (plan §6.6) over the committed eval JSONs.

Hero: terminal source gain vs optimizer steps T (log-x) — dose cells (filled)
vs positives-only twins (open), reused #601 references in gray, per-seed
points + seed-mean, frozen band shaded per pair; left panel log-prob, right
panel EOS margin.

Exploratory dump (over-produce; §6.6): dense growth trajectories per level
(dose vs twin, per seed); row-type CE channels (log scale) per dose cell;
mid-run per-step window zoom; capability trajectories; raw-alongside per-seed
terminal scatter (no banding).

Run AFTER scripts/i622_analyze.py (consumes its classification.json for the
artifact-locked margin band):
    uv run python scripts/i622_figures.py \
        [--slab-root eval_results/issue_622] [--fig-dir figures/issue_622]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("i622.figures")

SEEDS = (42, 137)
LEVELS = (
    ("16:1", "dose_200p3200n", "posonly_200p_T208"),
    ("32:1", "dose_200p6400n", "posonly_200p_T416"),
    ("64:1", "dose_200p12800n", "posonly_200p_T819"),
)
MIDRUN_WINDOWS = {  # 16-step per-step window per level (plan §4.2)
    "16:1": (104, 119),
    "32:1": (200, 215),
    "64:1": (400, 415),
}


def _terminal(traj: dict) -> dict:
    term = max(traj["checkpoints"], key=lambda c: c["frac"])
    ss = term["source_self"]
    dz_m = ss["z_marker_g_mean"] - ss["z_marker_b_mean"]
    dz_e = ss["z_eos_g_mean"] - ss["z_eos_b_mean"]
    return {
        "step": term.get("step"),
        "delta_g": float(ss["delta_g_mean"]),
        "delta_margin": float(dz_m - dz_e),
    }


def _load_units(dose_dir: Path) -> dict[str, dict]:
    units: dict[str, dict] = {}
    for _label, dose_slug, twin_slug in LEVELS:
        for slug in (dose_slug, twin_slug):
            for seed in SEEDS:
                cell_dir = dose_dir / f"{slug}_seed{seed}"
                if not (cell_dir / "trajectory.json").exists():
                    log.warning("missing trajectory for %s_seed%s — skipped", slug, seed)
                    continue
                unit = {
                    "terminal": _terminal(json.loads((cell_dir / "trajectory.json").read_text()))
                }
                for name in ("dense_trajectory", "rowtype_ce", "capability_trajectory"):
                    p = cell_dir / f"{name}.json"
                    if p.exists():
                        unit[name] = json.loads(p.read_text())
                units[f"{slug}_seed{seed}"] = unit
    return units


def main(argv: list[str] | None = None) -> int:  # noqa: C901 -- one block per registered figure
    ap = argparse.ArgumentParser(description="Task #622 figures (see module docstring).")
    ap.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_622"))
    ap.add_argument("--fig-dir", type=Path, default=Path("figures/issue_622"))
    args = ap.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s", stream=sys.stdout)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    dose_dir = args.slab_root / "dose_break"
    units = _load_units(dose_dir)
    cls_path = args.slab_root / "analysis" / "classification.json"
    cls = json.loads(cls_path.read_text()) if cls_path.exists() else None
    band_logprob = cls["frozen_bands"]["logprob_nats"] if cls else 5.58
    band_margin = cls["frozen_bands"]["margin_logits"] if cls else 2.18
    refs = cls["references_601"] if cls else {}

    c_dose = paper_palette_role("primary")
    c_twin = paper_palette_role("control")
    c_ref = paper_palette_role("neutral")

    # ── Hero: terminal gain vs T (log-x), two panels. ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, key, band, ylab in (
        (axes[0], "delta_g", band_logprob, "terminal source ΔlogP(' ※') [nats]"),
        (axes[1], "delta_margin", band_margin, "terminal source Δ(z_marker - z_eos) [logits]"),
    ):
        if key == "delta_g":
            for name, ref in refs.items():
                ax.scatter(
                    [ref["T"]] * len(ref["per_seed"]),
                    ref["per_seed"],
                    color=c_ref,
                    marker="s",
                    s=22,
                    zorder=2,
                    label="#601 references" if name == next(iter(refs)) else None,
                )
        for label, dose_slug, twin_slug in LEVELS:
            for slug, color, filled in ((dose_slug, c_dose, True), (twin_slug, c_twin, False)):
                pts = [
                    units[f"{slug}_seed{s}"]["terminal"]
                    for s in SEEDS
                    if f"{slug}_seed{s}" in units
                ]
                if not pts:
                    continue
                xs = [p["step"] for p in pts]
                ys = [p[key] for p in pts]
                ax.scatter(
                    xs,
                    ys,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    s=46,
                    zorder=3,
                    label=(
                        ("dose cells" if filled else "positives-only twins")
                        if label == "16:1"
                        else None
                    ),
                )
                mean_y = sum(ys) / len(ys)
                ax.scatter([sum(xs) / len(xs)], [mean_y], color=color, marker="_", s=180, zorder=4)
                if filled:
                    # frozen co-landing band shaded around the TWIN mean per pair
                    twin_pts = [
                        units[f"{twin_slug}_seed{s}"]["terminal"][key]
                        for s in SEEDS
                        if f"{twin_slug}_seed{s}" in units
                    ]
                    if twin_pts:
                        tm = sum(twin_pts) / len(twin_pts)
                        ax.axhspan(
                            tm - band,
                            tm + band,
                            xmin=0,
                            xmax=1,
                            color=c_twin,
                            alpha=0.07,
                            zorder=1,
                        )
        ax.set_xscale("log")
        ax.set_xlabel("optimizer steps T (log scale)")
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "hero_dose_to_failure", dir=args.fig_dir)
    plt.close(fig)

    # ── Dense growth trajectories per level (dose vs twin, per seed). ────────
    for label, dose_slug, twin_slug in LEVELS:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        for ax, key, ylab in (
            (axes[0], "delta_g", "source ΔlogP [nats]"),
            (axes[1], "delta_margin", "source Δ(z_marker - z_eos) [logits]"),
        ):
            for slug, color, ls in ((dose_slug, c_dose, "-"), (twin_slug, c_twin, "--")):
                for seed in SEEDS:
                    u = units.get(f"{slug}_seed{seed}")
                    if not u or "dense_trajectory" not in u:
                        continue
                    cks = sorted(
                        u["dense_trajectory"]["checkpoints"],
                        key=lambda c: (c["step"] is None, c["step"]),
                    )
                    ax.plot(
                        [c["step"] for c in cks],
                        [c["source_mean"][key] for c in cks],
                        color=color,
                        linestyle=ls,
                        alpha=0.85 if seed == 42 else 0.55,
                        label=f"{slug} seed{seed}",
                    )
            ax.set_xlabel("optimizer step")
            ax.set_ylabel(ylab)
            ax.legend(frameon=False, fontsize=7)
        savefig_paper(fig, f"dense_growth_{label.replace(':', 'to')}", dir=args.fig_dir)
        plt.close(fig)

        # Mid-run per-step window zoom (the sawtooth read).
        lo, hi = MIDRUN_WINDOWS[label]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for slug, color, ls in ((dose_slug, c_dose, "-"), (twin_slug, c_twin, "--")):
            for seed in SEEDS:
                u = units.get(f"{slug}_seed{seed}")
                if not u or "dense_trajectory" not in u:
                    continue
                cks = [
                    c
                    for c in u["dense_trajectory"]["checkpoints"]
                    if c["step"] is not None and lo - 1 <= c["step"] <= hi + 1
                ]
                cks.sort(key=lambda c: c["step"])
                if not cks:
                    continue
                ax.plot(
                    [c["step"] for c in cks],
                    [c["source_mean"]["delta_g"] for c in cks],
                    color=color,
                    linestyle=ls,
                    marker="o",
                    markersize=3,
                    alpha=0.85 if seed == 42 else 0.55,
                    label=f"{slug} seed{seed}",
                )
        ax.set_xlabel("optimizer step (per-step mid-run window)")
        ax.set_ylabel("source ΔlogP [nats]")
        ax.legend(frameon=False, fontsize=7)
        savefig_paper(fig, f"midrun_window_{label.replace(':', 'to')}", dir=args.fig_dir)
        plt.close(fig)

    # ── Row-type CE channels per dose cell (log scale). ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True)
    for ax, (label, dose_slug, _twin) in zip(axes, LEVELS, strict=True):
        for seed in SEEDS:
            u = units.get(f"{dose_slug}_seed{seed}")
            if not u or "rowtype_ce" not in u:
                continue
            rt = u["rowtype_ce"]
            ax.plot(
                rt["steps"],
                rt["pos_marker_ce"],
                color=c_dose,
                alpha=0.85 if seed == 42 else 0.55,
                label=f"pos seed{seed}",
            )
            ax.plot(
                rt["steps"],
                rt["neg_trailing_ce"],
                color=c_twin,
                alpha=0.85 if seed == 42 else 0.55,
                label=f"neg seed{seed}",
            )
        ax.axhline(1e-3, color=c_ref, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("optimizer step")
        ax.set_title(f"dose {label}")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("row-type CE [nats, log scale]")
    savefig_paper(fig, "rowtype_ce_channels", dir=args.fig_dir)
    plt.close(fig)

    # ── Capability trajectories (all units). ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for _label, dose_slug, twin_slug in LEVELS:
        for slug, color, ls in ((dose_slug, c_dose, "-"), (twin_slug, c_twin, "--")):
            for seed in SEEDS:
                u = units.get(f"{slug}_seed{seed}")
                if not u or "capability_trajectory" not in u:
                    continue
                recs = u["capability_trajectory"]["records"]
                if not recs:
                    continue
                ax.plot(
                    [r["step"] for r in recs],
                    [r["accuracy"] for r in recs],
                    color=color,
                    linestyle=ls,
                    alpha=0.5,
                    linewidth=1.0,
                )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("ARC-C logprob accuracy (200-q subsample)")
    savefig_paper(fig, "capability_trajectories", dir=args.fig_dir)
    plt.close(fig)

    # ── Raw-alongside: per-seed terminal scatter, no banding. ────────────────
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for _label, dose_slug, twin_slug in LEVELS:
        for slug, color, filled in ((dose_slug, c_dose, True), (twin_slug, c_twin, False)):
            for seed in SEEDS:
                u = units.get(f"{slug}_seed{seed}")
                if not u:
                    continue
                t = u["terminal"]
                ax.scatter(
                    [t["step"]],
                    [t["delta_g"]],
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    s=40,
                )
    ax.set_xscale("log")
    ax.set_xlabel("optimizer steps T (log scale)")
    ax.set_ylabel("terminal source ΔlogP [nats] — raw per-seed points")
    savefig_paper(fig, "terminal_scatter_raw", dir=args.fig_dir)
    plt.close(fig)

    log.info("figures written -> %s", args.fig_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
