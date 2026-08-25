#!/usr/bin/env python
"""Issue #1415 hooked-unhooked decomposition — VM figures (plan v11 §6, phase A3).

Reads the COMMITTED eval JSONs (git; produced by h3_stats + h1_fidelity) and
writes:

- HERO ``hooked_decomp_hero.png``: per-bin direct-component magnitude (log
  scale, the 2x same-text jitter band drawn) AND per-bin cosine-to-target
  (per-bin random-direction null p97.5 drawn) for the 4 primary cells
  (arm x steer layer at the REGISTERED headline read layers), with the
  round's realized-text divergence profile overlaid. Every curve label names
  quantity / layer / normalization.
- ``hooked_decomp_hero_normalized.png``: the ||diff||/||state|| variant
  beside the hero (per the Alternatives critic) — the direct component and
  jitter band divided per (pair, bin) by the unhooked state norm at the read
  layer; the realized-text overlay is OMITTED here (its state norms are not
  carried by the round-1 JSON — stated in the labels).
- The exploratory dump (over-produced by design, plan §6) under
  ``<out-dir>/hooked_decomp_explore/``: per-pair small multiples,
  per-read-layer overlays, steered-vs-baseline, per-draw dispersion fans,
  G1/G2 fidelity value distributions, matched-vs-cross overlays, and
  jitter-band anatomy.

No new statistics — every value plotted is a pure re-read of the committed
JSONs. Runs on the shared VM (trivial CPU; plan §9 A3 row).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE heavy imports — shared-VM thread caps bind in-process (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

BIN_ORDER = [
    "first",
    "tok2_5",
    *(f"dec{d}" for d in range(1, 11)),
    "last",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--results-root",
        default="eval_results/issue_1415/hooked_unhooked_decomposition",
        help="the committed h3_stats output root",
    )
    p.add_argument(
        "--overlay",
        default="eval_results/issue_1415/answer_position_shift_profile/per_pair_profiles.json",
        help="the round-1 realized-text per-pair profiles JSON (hero overlay)",
    )
    p.add_argument("--out-dir", default="figures/issue_1415")
    p.add_argument("--style", choices=("blog", "iclr"), default="blog")
    p.add_argument(
        "--skip-overlay",
        action="store_true",
        help="omit the realized-text overlay (tiny smoke against fixture outputs "
        "whose round-1 twin used different tiny cells)",
    )
    return p.parse_args(argv)


def _load(path: Path) -> dict:
    assert path.exists(), f"missing input JSON: {path}"
    return json.loads(path.read_text())


def _rows_for(rows: list[dict], label: str, read_layer: int) -> list[dict]:
    return [r for r in rows if r["label"] == label and r["read_layer"] == read_layer]


def _bin_mean(cell_rows: list[dict], field: str) -> np.ndarray:
    """Mean over pairs per bin (None-skipping; NaN where no pair defines it)."""
    out = []
    for b in BIN_ORDER:
        vals = [
            r2[field]
            for r in cell_rows
            for r2 in r["bins"]
            if r2["bin"] == b and r2.get(field) is not None
        ]
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.asarray(out)


def _bin_mean_normalized(cell_rows: list[dict], num: str, den: str) -> np.ndarray:
    """Mean over pairs per bin of num/den (per-pair ratio first, then mean)."""
    out = []
    for b in BIN_ORDER:
        vals = []
        for r in cell_rows:
            for r2 in r["bins"]:
                if r2["bin"] == b and r2.get(num) is not None and r2.get(den):
                    vals.append(r2[num] / r2[den])
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.asarray(out)


def _overlay_mag(overlay_rows: list[dict], arm: str, steer_layer: int) -> np.ndarray:
    rows = [
        r
        for r in overlay_rows
        if r.get("round") == "primary" and r["arm"] == arm and r["steer_layer"] == steer_layer
    ]
    out = []
    for b in BIN_ORDER:
        vals = [
            r2["magnitude"]
            for r in rows
            for r2 in r["bins"]
            if r2["bin"] == b and r2.get("magnitude") is not None
        ]
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.asarray(out)


def _primary_cells(summary: dict) -> list[tuple[str, str, int, int]]:
    """(label, arm, steer_layer, headline_read_layer) for the 4 lattice cells."""
    cells = []
    for label, info in sorted(summary["labels"].items()):
        if info.get("lattice"):
            cells.append(
                (label, info["delta_arm"], info["steer_layer"], info["headline_read_layer"])
            )
    assert cells, "no lattice cells in summary.json"
    return cells


def _style_bin_axis(ax) -> None:
    ax.set_xticks(range(len(BIN_ORDER)))
    ax.set_xticklabels(BIN_ORDER, rotation=60, ha="right", fontsize=6)


def hero(
    rows: list[dict],
    summary: dict,
    overlay_rows: list[dict] | None,
    out_path: Path,
    normalized: bool,
) -> None:
    cells = _primary_cells(summary)
    fig, axes = plt.subplots(
        2, len(cells), figsize=(3.1 * len(cells), 5.6), layout="constrained", sharex=True
    )
    x = np.arange(len(BIN_ORDER))
    for col, (label, arm, steer, read) in enumerate(cells):
        cell_rows = _rows_for(rows, label, read)
        ax = axes[0][col]
        if normalized:
            mag = _bin_mean_normalized(cell_rows, "magnitude", "unhooked_norm_read")
            jit = _bin_mean_normalized(cell_rows, "jitter", "unhooked_norm_read")
            mag_lab = f"direct ‖hooked-unhooked‖/‖state‖ (read L{read})"
            jit_lab = f"2× same-text jitter /‖state‖ (read L{read})"
        else:
            mag = _bin_mean(cell_rows, "magnitude")
            jit = _bin_mean(cell_rows, "jitter")
            mag_lab = f"direct ‖hooked-unhooked‖ (raw L2, read L{read})"
            jit_lab = f"2× same-text jitter (raw L2, read L{read})"
        ax.plot(x, mag, marker="o", ms=3, color=paper_palette_role("primary"), label=mag_lab)
        ax.fill_between(
            x,
            np.zeros_like(jit),
            2.0 * jit,
            color=paper_palette_role("neutral"),
            alpha=0.35,
            label=jit_lab,
        )
        if overlay_rows is not None and not normalized:
            ov = _overlay_mag(overlay_rows, arm, steer)
            ax.plot(
                x,
                ov,
                marker="s",
                ms=3,
                ls="--",
                color=paper_palette_role("baseline"),
                label=f"realized text ‖steered-baseline‖ (raw L2, read L{steer}=steer)",
            )
        ax.set_yscale("log")
        ax.set_title(f"{arm} arm, steer L{steer} → read L{read}", fontsize=8)
        if col == 0:
            ax.set_ylabel("‖Δstate‖/‖state‖ per bin" if normalized else "‖Δstate‖ per bin (log)")
        ax.legend(fontsize=5, loc="lower left")

        ax2 = axes[1][col]
        cos = _bin_mean(cell_rows, "alignment")
        null = _bin_mean(cell_rows, "null_p975")
        ax2.plot(
            x,
            cos,
            marker="o",
            ms=3,
            color=paper_palette_role("accent"),
            label=f"cos(direct diff, target) (read L{read})",
        )
        ax2.plot(
            x,
            null,
            ls=":",
            color=paper_palette_role("control"),
            label="random-direction null p97.5 (seed 14154)",
        )
        ax2.axhline(0.0, lw=0.6, color="0.6")
        ax2.set_ylim(-1.05, 1.05)
        if col == 0:
            ax2.set_ylabel("cosine to counterfactual target")
        _style_bin_axis(ax2)
        ax2.legend(fontsize=5, loc="lower left")
    fig.suptitle(
        "Hooked-unhooked direct component per answer bin (mean over pairs; "
        "headline read layers pre-registered above steer)",
        fontsize=9,
    )
    savefig_paper(fig, out_path)
    plt.close(fig)


def explore(rows: list[dict], summary: dict, fidelity: dict | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(BIN_ORDER))
    cells = _primary_cells(summary)

    # (1) per-pair small multiples (magnitude, headline read layer).
    for label, arm, steer, read in cells:
        cell_rows = _rows_for(rows, label, read)
        fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
        colors = paper_palette(max(len(cell_rows), 3))
        for i, r in enumerate(sorted(cell_rows, key=lambda r: r["pair_id"])):
            mags = [
                next((b2["magnitude"] for b2 in r["bins"] if b2["bin"] == b), None)
                for b in BIN_ORDER
            ]
            y = np.asarray([np.nan if m is None else m for m in mags])
            flags = [k for k, v in r["flags"].items() if v]
            ax.plot(
                x,
                y,
                lw=0.8,
                alpha=0.8,
                color=colors[i % len(colors)],
                label=f"{r['pair_id']}" + (f" [{','.join(flags)}]" if flags else ""),
            )
        ax.set_yscale("log")
        ax.set_title(f"per-pair direct magnitude — {label} (read L{read})", fontsize=8)
        _style_bin_axis(ax)
        ax.legend(fontsize=4, ncols=2, loc="upper right")
        savefig_paper(fig, out_dir / f"perpair_mag_{label.replace('/', '_')}.png")
        plt.close(fig)

    # (2) per-read-layer overlays (every above-steer layer, mean over pairs).
    for label, info in sorted(summary["labels"].items()):
        read_layers = sorted(int(k) for k in info["read_layers"])
        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        colors = paper_palette(max(len(read_layers), 3))
        for i, rl in enumerate(read_layers):
            mag = _bin_mean(_rows_for(rows, label, rl), "magnitude")
            hl = " (headline)" if info["read_layers"][str(rl)]["headline"] else ""
            ax.plot(x, mag, marker="o", ms=2.5, color=colors[i], label=f"read L{rl}{hl}")
        ax.set_yscale("log")
        ax.set_title(f"direct magnitude by read layer — {label}", fontsize=8)
        _style_bin_axis(ax)
        ax.legend(fontsize=6)
        savefig_paper(fig, out_dir / f"readlayer_overlay_{label.replace('/', '_')}.png")
        plt.close(fig)

    # (3) steered-text vs baseline-text direct component (context arm).
    for lkey, block in sorted(summary["steered_vs_baseline"].items()):
        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        s = np.asarray(
            [
                (block["bins"][b]["steered_mag_mean"] or np.nan) if b in block["bins"] else np.nan
                for b in BIN_ORDER
            ],
            dtype=float,
        )
        bmag = np.asarray(
            [
                (block["bins"][b]["baseline_mag_mean"] or np.nan) if b in block["bins"] else np.nan
                for b in BIN_ORDER
            ],
            dtype=float,
        )
        ax.plot(
            x,
            s,
            marker="o",
            ms=3,
            color=paper_palette_role("primary"),
            label=f"steered text (context arm, read L{block['read_layer']})",
        )
        ax.plot(
            x,
            bmag,
            marker="s",
            ms=3,
            ls="--",
            color=paper_palette_role("control"),
            label=f"baseline text (pure direct, read L{block['read_layer']})",
        )
        ax.set_yscale("log")
        ax.set_title(f"steered vs baseline text direct component — steer {lkey}", fontsize=8)
        _style_bin_axis(ax)
        ax.legend(fontsize=6)
        savefig_paper(fig, out_dir / f"steered_vs_baseline_{lkey}.png")
        plt.close(fig)

    # (4) per-draw dispersion fans (headline read layer, min/mean/max over draws).
    for label, arm, steer, read in cells:
        cell_rows = _rows_for(rows, label, read)
        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        lo, mid, hi = [], [], []
        for b in BIN_ORDER:
            vals: list[float] = []
            for r in cell_rows:
                for b2 in r["bins"]:
                    if b2["bin"] == b:
                        vals.extend(v for v in b2["per_draw_magnitudes"] if v is not None)
            lo.append(min(vals) if vals else np.nan)
            hi.append(max(vals) if vals else np.nan)
            mid.append(float(np.mean(vals)) if vals else np.nan)
        ax.fill_between(
            x,
            lo,
            hi,
            alpha=0.3,
            color=paper_palette_role("neutral"),
            label="per-draw min-max across pairs",
        )
        ax.plot(
            x, mid, marker="o", ms=3, color=paper_palette_role("primary"), label="per-draw mean"
        )
        ax.set_yscale("log")
        ax.set_title(f"per-draw dispersion — {label} (read L{read})", fontsize=8)
        _style_bin_axis(ax)
        ax.legend(fontsize=6)
        savefig_paper(fig, out_dir / f"dispersion_{label.replace('/', '_')}.png")
        plt.close(fig)

    # (5) G1/G2 fidelity value distributions.
    if fidelity is not None:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0), layout="constrained")
        g1 = [c["min_cos"] for c in fidelity["g1"]["cells"]]
        a1.hist(g1, bins=20, color=paper_palette_role("primary"))
        a1.axvline(fidelity["g1"]["halt_cos"], ls="--", color=paper_palette_role("control"))
        a1.set_title("G1 upstream-zero min cosine per spot cell", fontsize=8)
        g2c = [c["cos_d_delta"] for c in fidelity["g2"]["cells"]]
        g2r = [c["norm_ratio"] for c in fidelity["g2"]["cells"]]
        a2.scatter(g2r, g2c, s=12, color=paper_palette_role("accent"))
        a2.axhline(fidelity["g2"]["cos_min"], ls="--", color=paper_palette_role("control"))
        for xr in fidelity["g2"]["norm_ratio_band"]:
            a2.axvline(xr, ls=":", color=paper_palette_role("control"))
        a2.set_xlabel("‖d‖ / (alpha*‖Δ‖)")
        a2.set_ylabel("cos(d, Δ)")
        a2.set_title("G2 edit-injection exactness per spot cell", fontsize=8)
        savefig_paper(fig, out_dir / "fidelity_values.png")
        plt.close(fig)

    # (6) matched-vs-cross overlays (headline read layer).
    for label, arm, steer, read in cells:
        cell_rows = _rows_for(rows, label, read)
        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        for ptype, role in (("matched", "primary"), ("cross", "accent")):
            sub = [r for r in cell_rows if r["pair_type"] == ptype]
            mag = _bin_mean(sub, "magnitude")
            ax.plot(
                x,
                mag,
                marker="o",
                ms=3,
                color=paper_palette_role(role),
                label=f"{ptype} pairs (n={len(sub)})",
            )
        ax.set_yscale("log")
        ax.set_title(f"matched vs cross — {label} (read L{read})", fontsize=8)
        _style_bin_axis(ax)
        ax.legend(fontsize=6)
        savefig_paper(fig, out_dir / f"matched_vs_cross_{label.replace('/', '_')}.png")
        plt.close(fig)

    # (7) jitter-band anatomy (empirical transported vs fp16 floor).
    for label, arm, steer, read in cells:
        cell_rows = _rows_for(rows, label, read)
        fig, ax = plt.subplots(figsize=(4.6, 3.2), layout="constrained")
        emp = _bin_mean(cell_rows, "jitter_empirical_transported")
        flo = _bin_mean(cell_rows, "jitter_fp16_floor")
        ax.plot(
            x,
            emp,
            marker="o",
            ms=3,
            color=paper_palette_role("primary"),
            label=f"empirical jitter transported L{steer}→L{read}",
        )
        ax.plot(
            x,
            flo,
            marker="s",
            ms=3,
            ls="--",
            color=paper_palette_role("neutral"),
            label="fp16 quantization floor (2⁻¹¹·√2·‖state‖)",
        )
        ax.set_yscale("log")
        ax.set_title(f"jitter-band anatomy — {label}", fontsize=8)
        _style_bin_axis(ax)
        ax.legend(fontsize=6)
        savefig_paper(fig, out_dir / f"jitter_anatomy_{label.replace('/', '_')}.png")
        plt.close(fig)


def hero_iclr(rows: list[dict], summary: dict, out_dir: Path) -> None:
    """ICLR paper variant: normalized direct component per answer bin, 2x2 grid.

    One panel per primary lattice cell; blue line = re-forwarded direct
    component as a fraction of the state norm (log scale), gray band = 2x the
    same-text jitter floor. The cosine row of the blog hero moves to prose.
    """
    from explore_persona_space.analysis.paper_plots import figsize_iclr_panels, paper_color

    cells = _primary_cells(summary)
    assert len(cells) == 4, [c[0] for c in cells]
    fig, axes = plt.subplots(
        2, 2, figsize=figsize_iclr_panels(2, height_in=3.6), sharex=True, sharey=True
    )
    x = np.arange(len(BIN_ORDER))
    tick_idx = [0, 1, 3, 5, 7, 9, 11, 12]
    tick_labels = ["first", "tok 2–5", "20%", "40%", "60%", "80%", "100%", "last"]
    for ax, (label, arm, steer, read) in zip(axes.ravel(), cells, strict=True):
        cell_rows = _rows_for(rows, label, read)
        mag = _bin_mean_normalized(cell_rows, "magnitude", "unhooked_norm_read")
        jit = _bin_mean_normalized(cell_rows, "jitter", "unhooked_norm_read")
        ax.plot(x, mag, marker="o", ms=2.5, color=paper_color("instruct"), lw=1.2)
        ax.fill_between(x, np.zeros_like(jit), 2.0 * jit, color=paper_color("null"), alpha=0.35)
        ax.set_yscale("log")
        ax.set_title(f"{arm} arm, patch L{steer} → read L{read}")
        ax.set_xticks(tick_idx, tick_labels, rotation=45, ha="right", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("‖direct Δstate‖ / ‖state‖")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    axes[0, 0].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ms=2.5,
                lw=1.2,
                color=paper_color("instruct"),
                label="re-forwarded direct component",
            ),
            Patch(facecolor=paper_color("null"), alpha=0.35, label="2× same-text jitter"),
        ],
        fontsize=7,
        loc="upper right",
    )
    fig.tight_layout()
    savefig_paper(fig, "c4_patch_persistence", dir=out_dir)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.style == "iclr":
        set_paper_style("iclr")
        root = Path(args.results_root)
        per_pair = _load(root / "per_pair_direct_profiles.json")
        summary = _load(root / "summary.json")
        out_dir = Path("figures/paper")
        out_dir.mkdir(parents=True, exist_ok=True)
        hero_iclr(per_pair["rows"], summary, out_dir)
        print(f"wrote iclr hero under {out_dir}")
        return
    set_paper_style()
    root = Path(args.results_root)
    per_pair = _load(root / "per_pair_direct_profiles.json")
    summary = _load(root / "summary.json")
    fid_path = root / "fidelity_gate_report.json"
    fidelity = json.loads(fid_path.read_text()) if fid_path.exists() else None
    rows = per_pair["rows"]
    overlay_rows: list[dict] | None = None
    if not args.skip_overlay:
        overlay_rows = _load(Path(args.overlay))["profiles"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hero(rows, summary, overlay_rows, out_dir / "hooked_decomp_hero.png", normalized=False)
    hero(rows, summary, None, out_dir / "hooked_decomp_hero_normalized.png", normalized=True)
    explore(rows, summary, fidelity, out_dir / "hooked_decomp_explore")
    print(f"wrote hero + normalized hero + exploratory dump under {out_dir}")


if __name__ == "__main__":
    main()
