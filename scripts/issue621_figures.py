"""Issue #621 figures (VM CPU) — heroes + exploratory panels from analysis.json.

Heroes (plan §6):
  1. ``read_identity_profile`` — |cos(â, v̂_c_src)| layer profile per arm at
     the primary position, with the wrong-context null p95 band and the
     a_init overlay (the H1/H2 picture in one panel).
  2. ``firing_vs_leakage`` — predicted firing vs measured Δlog P per
     bystander, one panel per arm, per-cell median ρ + comparator inset.

Exploratory:
  3. ``cross_seed_strip`` — cross-seed |cos| of a vs b per (arm, source),
     with #604's 0.015 / 0.93 reference lines.
  4. ``wu_profile`` — cos(b̂, Ŵ_U[※]) + EOS-margin layer profiles (write
     arm) with the matched max-selection null p95.
  5. ``h4_position_sweep`` — median ρ per arm × capture position.

All figures via paper_plots (set_paper_style + savefig_paper → PNG/PDF +
meta.json with commit pin).

CLI:
    uv run python scripts/issue621_figures.py \\
        [--analysis eval_results/issue_621/analysis/analysis.json] \\
        [--out-dir figures/issue_621]
"""

# ruff: noqa: RUF001, RUF002, RUF003  # math notation in labels

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

log = logging.getLogger("issue_621.figures")

PRIMARY_POSITION = "end_of_response"
ARM_ORDER = ("read", "write", "bridge")
# Cross-seed reference lines from #604 (top-1 key 0.015; write 0.927).
REF_KEY_604 = 0.015
REF_WRITE_604 = 0.927


def _arm_cells(cells: dict, arm: str) -> list[dict]:
    return [c for c in cells.values() if c["arm"] == arm]


def fig_read_identity_profile(cells: dict, out_dir: Path) -> None:
    """Hero 1: per-arm layer profile of |cos(â, v̂_src)| vs null + a_init."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, arm in zip(axes, ARM_ORDER, strict=True):
        arm_cells = _arm_cells(cells, arm)
        if not arm_cells:
            ax.set_visible(False)
            continue
        src_profiles, null_profiles, init_profiles = [], [], []
        for c in arm_cells:
            spaces = c["read_identity"][PRIMARY_POSITION]
            for space, per_module in spaces.items():
                if space == "raw_gamma":
                    continue
                for _module, block in per_module.items():
                    src_profiles.append(block["cos_src_per_layer"])
                    null_profiles.append(block["null_p95_per_layer"])
            for _module, block in c["a_init"].items():
                init_profiles.append(block["cos_a_init_per_layer"])
        layers = np.arange(len(src_profiles[0]))
        src = np.nanmean(np.asarray(src_profiles, dtype=float), axis=0)
        null = np.nanmean(np.asarray(null_profiles, dtype=float), axis=0)
        ax.plot(layers, src, color=paper_palette_role("primary"), label="|cos(â, v̂_src)|")
        ax.fill_between(
            layers,
            0,
            null,
            color=paper_palette_role("neutral"),
            alpha=0.35,
            label="wrong-context null p95",
        )
        if init_profiles:
            init = np.nanmean(np.asarray(init_profiles, dtype=float), axis=0)
            ax.plot(
                layers,
                init,
                color=paper_palette_role("control"),
                linestyle="--",
                label="|cos(a_t, a_init)|",
            )
        ax.axvspan(14, 24, color=paper_palette_role("baseline"), alpha=0.08)
        ax.set_title(f"{arm} arm (n={len(arm_cells)} cells)")
        ax.set_xlabel("layer")
    axes[0].set_ylabel("|cosine|")
    axes[0].legend(fontsize=7, loc="upper left")
    savefig_paper(fig, out_dir / "read_identity_profile")


def fig_firing_vs_leakage(cells: dict, out_dir: Path) -> None:
    """Hero 2: firing vs Δlog P scatter per arm + comparator-median inset."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, arm in zip(axes, ARM_ORDER, strict=True):
        arm_cells = _arm_cells(cells, arm)
        if not arm_cells:
            ax.set_visible(False)
            continue
        mode = "write" if arm in ("write", "bridge") else "abs"
        dv_key = "rho_vs_delta_margin" if mode == "write" else "rho_vs_delta_logp"
        rhos = []
        colors = paper_palette(len(arm_cells))
        for ci, c in enumerate(arm_cells):
            block = c["h4"][PRIMARY_POSITION].get(mode) or c["h4"][PRIMARY_POSITION]["abs"]
            firing = block["firing"]
            dv = (
                c["dv_per_bystander"]["delta_margin"]
                if mode == "write"
                else c["dv_per_bystander"]["delta_logp"]
            )
            bys = c["primary_bystanders"]
            xs = [firing[p] for p in bys]
            ys = [dv[p] for p in bys]
            ax.scatter(xs, ys, s=10, alpha=0.55, color=colors[ci])
            rho = block["primary_excl_trained_negs"][dv_key]
            if not (rho is None or np.isnan(rho)):
                rhos.append(rho)
        med = float(np.median(rhos)) if rhos else float("nan")
        ax.set_title(f"{arm}: median ρ = {med:+.2f} (n={len(rhos)})")
        ax.set_xlabel("predicted firing")
        ax.set_ylabel("Δ log P(※)" if mode != "write" else "Δ(z※ − z_eos)")
        # Comparator inset (median ρ across cells).
        cmp_meds: dict[str, float] = {}
        for c in arm_cells:
            block = c["h4"][PRIMARY_POSITION].get(mode) or c["h4"][PRIMARY_POSITION]["abs"]
            for name, cb in block["comparators_primary"].items():
                v = cb[dv_key]
                if v is not None and not np.isnan(v):
                    cmp_meds.setdefault(name, []).append(v)
        if cmp_meds:
            ins = ax.inset_axes([0.62, 0.06, 0.36, 0.32])
            names = sorted(cmp_meds)
            vals = [float(np.median(cmp_meds[n])) for n in names]
            ins.barh(range(len(names)), vals, color=paper_palette_role("neutral"), height=0.6)
            ins.axvline(med, color=paper_palette_role("accent"), lw=1)
            ins.set_yticks(range(len(names)))
            ins.set_yticklabels([n.replace("_", " ") for n in names], fontsize=5)
            ins.tick_params(axis="x", labelsize=5)
    savefig_paper(fig, out_dir / "firing_vs_leakage")


def fig_cross_seed_strip(summary: dict, out_dir: Path) -> None:
    """Exploratory: cross-seed |cos| of a vs b, with #604 reference lines."""
    import matplotlib.pyplot as plt

    cs = summary["cross_seed"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    keys = sorted(cs)
    xs = np.arange(len(keys))
    a_vals = [cs[k]["band_mean_abs_cos_a"] for k in keys]
    b_vals = [cs[k]["band_mean_abs_cos_b"] for k in keys]
    ax.scatter(xs - 0.12, a_vals, color=paper_palette_role("primary"), label="a (read)", s=24)
    ax.scatter(xs + 0.12, b_vals, color=paper_palette_role("accent"), label="b (write)", s=24)
    ax.axhline(REF_KEY_604, color=paper_palette_role("neutral"), ls=":", lw=1)
    ax.axhline(REF_WRITE_604, color=paper_palette_role("neutral"), ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([k.replace("|", "\n") for k in keys], fontsize=6)
    ax.set_ylabel("cross-seed |cos| (band mean L14–24)")
    ax.legend(fontsize=7)
    savefig_paper(fig, out_dir / "cross_seed_strip")


def fig_wu_profile(cells: dict, out_dir: Path) -> None:
    """Exploratory: W_U[※] + EOS-margin layer profiles on the write arm."""
    import matplotlib.pyplot as plt

    write_cells = [c for c in cells.values() if c["write_identity"]]
    if not write_cells:
        log.warning("no write-identity cells — skipping wu_profile")
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    wu_profiles, margin_profiles, null_p95s = [], [], []
    for c in write_cells:
        for _module, block in c["write_identity"].items():
            wu_profiles.append(block["cos_wu_marker_per_layer"])
            margin_profiles.append(block["cos_wu_eos_margin_per_layer"])
            if block.get("null_max_p95_norm_matched") is not None:
                null_p95s.append(block["null_max_p95_norm_matched"])
    layers = np.arange(len(wu_profiles[0]))
    ax.plot(
        layers,
        np.nanmean(np.asarray(wu_profiles, dtype=float), axis=0),
        color=paper_palette_role("primary"),
        label="cos(b̂, Ŵ_U[※])",
    )
    ax.plot(
        layers,
        np.nanmean(np.asarray(margin_profiles, dtype=float), axis=0),
        color=paper_palette_role("accent"),
        ls="--",
        label="cos(b̂, Ŵ_U[※]−Ŵ_U[eos])",
    )
    if null_p95s:
        ax.axhline(
            float(np.mean(null_p95s)),
            color=paper_palette_role("neutral"),
            ls=":",
            label="matched-max wrong-token null p95",
        )
    ax.axvspan(20, 27, color=paper_palette_role("baseline"), alpha=0.08)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine")
    ax.legend(fontsize=7)
    savefig_paper(fig, out_dir / "wu_profile")


def fig_h4_position_sweep(cells: dict, out_dir: Path) -> None:
    """Exploratory: median ρ per arm × capture position (duty 7 sensitivity)."""
    import matplotlib.pyplot as plt

    positions = ("end_of_prompt", "response_mean", "end_of_response")
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.25
    colors = paper_palette(len(ARM_ORDER))
    for ai, arm in enumerate(ARM_ORDER):
        arm_cells = _arm_cells(cells, arm)
        if not arm_cells:
            continue
        mode = "write" if arm in ("write", "bridge") else "abs"
        dv_key = "rho_vs_delta_margin" if mode == "write" else "rho_vs_delta_logp"
        meds = []
        for pos in positions:
            rhos = []
            for c in arm_cells:
                block = c["h4"][pos].get(mode) or c["h4"][pos]["abs"]
                v = block["primary_excl_trained_negs"][dv_key]
                if v is not None and not np.isnan(v):
                    rhos.append(v)
            meds.append(float(np.median(rhos)) if rhos else np.nan)
        ax.bar(
            np.arange(len(positions)) + (ai - 1) * width,
            meds,
            width=width,
            color=colors[ai],
            label=arm,
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([p.replace("_", " ") for p in positions])
    ax.set_ylabel("median per-cell ρ (primary set)")
    ax.legend(fontsize=7)
    savefig_paper(fig, out_dir / "h4_position_sweep")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis", default="eval_results/issue_621/analysis/analysis.json")
    ap.add_argument("--out-dir", default="figures/issue_621")
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.analysis).read_text())
    cells = payload["cells"]
    summary = payload["summary"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    fig_read_identity_profile(cells, out_dir)
    fig_firing_vs_leakage(cells, out_dir)
    fig_cross_seed_strip(summary, out_dir)
    fig_wu_profile(cells, out_dir)
    fig_h4_position_sweep(cells, out_dir)
    log.info("figures written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
