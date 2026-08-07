#!/usr/bin/env python3
"""Issue #2162 — manifest-driven figure set (plan §6 report figures).

Consumes ONLY the analysis outputs under ``eval_results/issue_2162/f_metrics/``
and renders to ``figures/issue_2162/``:

1. ``hero_ftype``        — per (cell x slot) 3-bar steered/shuffled/crosstype
                           mean F_beh + pair-clustered bootstrap 95% CIs,
                           grouped by family; companion: per-pair steered
                           F_beh points (the low-level per-unit view).
2. ``two_by_two``        — probe max-AUC (read) x steered F_beh (write)
                           scatter, causal verdict coded incl.
                           ``untestable-causal``.
3. ``layer_profile``     — probe AUC-per-layer heatmap, one panel per slot.
4. ``route_contrasts``   — P2 route variants beside their base types.
5. ``dose_position``     — recency-depth / load curves vs each crossed
                           cell's base type.
6. ``margin_validation`` — rho(margin shift, F_beh) scatter (rule-19 read).
7. ``diagnostics``       — per-cell coherence + cap-hit + post-exclusion n.

Every PNG gets a ``.meta.json`` sidecar (inputs + git provenance). Errorbar
offsets are non-negative by construction (the xerr/yerr gotcha).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2162.figures")

FAMILY_COLORS = {"P1": "#4878d0", "P2": "#ee854a", "P3": "#6acc64", None: "#9d9d9d"}
ARM_COLORS = {"steered": "#4878d0", "shuffled": "#9d9d9d", "crosstype": "#c44e52"}


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _save(fig, out_dir: Path, name: str, inputs: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    meta = {
        "figure": name,
        "inputs": [str(p) for p in inputs],
        **as_metadata_dict(git_provenance()),
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("[figures] wrote %s", png)


def _err(lo: float | None, hi: float | None, v: float) -> tuple[float, float]:
    """Non-negative errorbar offsets from CI bounds (NaN when absent)."""
    if lo is None or hi is None:
        return float("nan"), float("nan")
    return max(0.0, v - lo), max(0.0, hi - v)


def _cells_sorted(per_cell: dict) -> list[tuple[str, dict]]:
    fam_rank = {"P1": 0, "P2": 1, "P3": 2, None: 3}
    return sorted(
        per_cell.items(),
        key=lambda kv: (fam_rank.get(kv[1]["family"], 3), kv[1]["cell"], kv[1]["slot"]),
    )


def fig_hero(stats: dict, f_cells: list[dict], out_dir: Path, inputs: list[Path]) -> None:
    items = _cells_sorted(stats["per_cell"])
    labels = [f"{r['cell']}|{r['slot']}" for _, r in items]
    x = np.arange(len(items))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(14, len(items) * 0.32), 6))
    for k, arm in enumerate(("steered", "shuffled", "crosstype")):
        vals, lo_off, hi_off = [], [], []
        for _, r in items:
            v = r.get(f"f_{arm}_mean")
            ci = (r.get("ci95") or {}).get(arm, [None, None])
            vals.append(np.nan if v is None else v)
            e = _err(ci[0], ci[1], v if v is not None else 0.0)
            lo_off.append(e[0])
            hi_off.append(e[1])
        ax.bar(
            x + (k - 1) * width,
            vals,
            width,
            yerr=[lo_off, hi_off],
            color=ARM_COLORS[arm],
            label=arm,
            error_kw={"lw": 0.7},
        )
    for i, (_, r) in enumerate(items):
        if r["untestable_causal"]:
            ax.text(x[i], 0.02, "n/a", ha="center", fontsize=5, rotation=90, color="#555")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("F_beh (mean over post-exclusion pairs)")
    ax.set_title("Per-(type-cell x slot) F_beh: steered vs both nulls (95% pair-clustered CIs)")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, out_dir, "hero_ftype", inputs)

    # Per-unit companion: per-pair steered F_beh points.
    by_key: dict[str, list[float]] = defaultdict(list)
    for r in f_cells:
        if r["f_beh"] is not None:
            by_key[f"{r['cell']}|{r['slot']}"].append(r["f_beh"])
    fig, ax = plt.subplots(figsize=(max(14, len(items) * 0.32), 6))
    for i, lab in enumerate(labels):
        ys = by_key.get(lab, [])
        fam = items[i][1]["family"]
        ax.scatter([i] * len(ys), ys, s=6, alpha=0.55, color=FAMILY_COLORS.get(fam, "#999"))
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_ylabel("per-pair F_beh (steered)")
    ax.set_title("Per-pair steered F_beh (every pair, pre-aggregation; family-colored)")
    _save(fig, out_dir, "hero_ftype_perpair", inputs)


def fig_two_by_two(two: dict, out_dir: Path, inputs: list[Path]) -> None:
    rows = two["cells"]
    fig, ax = plt.subplots(figsize=(9, 8))
    style = {
        "positive": ("o", "#2a9d2a"),
        "null": ("o", "#9d9d9d"),
        "untestable-causal": ("x", "#c44e52"),
    }
    for verdict, (marker, color) in style.items():
        pts = [r for r in rows if r["causal_verdict"] == verdict and r["max_auc"] is not None]
        ax.scatter(
            [r["max_auc"] for r in pts],
            [r["f_steered_mean"] if r["f_steered_mean"] is not None else 0.0 for r in pts],
            marker=marker,
            s=26,
            color=color,
            label=f"causal: {verdict} (n={len(pts)})",
            alpha=0.85,
        )
        for r in pts:
            ax.annotate(
                f"{r['cell']}|{r['slot']}",
                (r["max_auc"], r["f_steered_mean"] or 0.0),
                fontsize=4.5,
                alpha=0.8,
            )
    ax.axvline(0.5, color="k", lw=0.6, ls=":")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("probe max-over-layers AUC (read)")
    ax.set_ylabel("steered F_beh mean (write)")
    ax.set_title("Read x write 2x2 (every cell x slot; untestable-causal marked x)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "two_by_two", inputs)


def fig_layer_profile(probe: dict, out_dir: Path, inputs: list[Path]) -> None:
    results = probe["results"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    for ax, slot in zip(axes, ("ce", "pe"), strict=True):
        rows = [r for r in results if r["slot"] == slot]
        rows.sort(key=lambda r: r["cell"])
        mat = np.array([r["auc_per_layer"] for r in rows])
        im = ax.imshow(mat, aspect="auto", vmin=0.3, vmax=1.0, cmap="viridis")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r["cell"] for r in rows], fontsize=5.5)
        ax.set_xlabel("layer")
        ax.set_title(f"probe AUC per layer — slot {slot}")
    fig.colorbar(im, ax=axes, shrink=0.7, label="LOCO AUC")
    _save(fig, out_dir, "layer_profile", inputs)


def fig_route_contrasts(stats: dict, out_dir: Path, inputs: list[Path]) -> None:
    from explore_persona_space.experiments.issue2162 import bank2162 as B

    per_cell = stats["per_cell"]
    route_pairs = [
        ("instr_format", "demo_format"),
        ("persona_prompted", "demo_persona"),
        ("instr_language", "language_implied"),
        ("persona_prompted", "persona_role_header"),
    ]
    conflict = [c for c in B.all_cells() if c.startswith("conflict_")]
    groups = [(a, b) for a, b in route_pairs] + [(B.base_type_of(c), c) for c in conflict]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = 0.0
    ticks, tick_labels = [], []
    for base, variant in groups:
        for slot in ("ce", "pe"):
            for off, cell, color in ((0.0, base, "#4878d0"), (0.35, variant, "#ee854a")):
                r = per_cell.get(f"{cell}|{slot}")
                if r is None or r.get("f_steered_mean") is None:
                    continue
                ci = (r.get("ci95") or {}).get("steered", [None, None])
                lo, hi = _err(ci[0], ci[1], r["f_steered_mean"])
                ax.bar(
                    x + off,
                    r["f_steered_mean"],
                    0.32,
                    yerr=[[lo], [hi]],
                    color=color,
                    error_kw={"lw": 0.7},
                )
            ticks.append(x + 0.17)
            tick_labels.append(f"{variant}|{slot}")
            x += 1.0
        x += 0.5
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_ylabel("steered F_beh")
    ax.set_title("P2 route contrasts: base type (blue) vs route variant / conflict (orange)")
    _save(fig, out_dir, "route_contrasts", inputs)


def fig_dose_position(stats: dict, out_dir: Path, inputs: list[Path]) -> None:
    from explore_persona_space.experiments.issue2162 import bank2162 as B

    per_cell = stats["per_cell"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, prefix, xlab in zip(
        axes, ("recency", "load"), ("history depth d", "distractor load l"), strict=True
    ):
        bases = sorted({B.base_type_of(c) for c in B.crossed_cells() if c.startswith(prefix)})
        for base in bases:
            for slot, ls in (("ce", "-"), ("pe", "--")):
                xs, ys, lo_off, hi_off = [], [], [], []
                base_rec = per_cell.get(f"{base}|{slot}")
                if base_rec and base_rec.get("f_steered_mean") is not None:
                    xs.append(0)
                    ys.append(base_rec["f_steered_mean"])
                    ci = (base_rec.get("ci95") or {}).get("steered", [None, None])
                    e = _err(ci[0], ci[1], ys[-1])
                    lo_off.append(e[0])
                    hi_off.append(e[1])
                depth_tag = "d" if prefix == "recency" else "l"
                for depth in (3, 5):
                    r = per_cell.get(f"{prefix}_{base}_{depth_tag}{depth}|{slot}")
                    if r is None or r.get("f_steered_mean") is None:
                        continue
                    xs.append(depth)
                    ys.append(r["f_steered_mean"])
                    ci = (r.get("ci95") or {}).get("steered", [None, None])
                    e = _err(ci[0], ci[1], ys[-1])
                    lo_off.append(e[0])
                    hi_off.append(e[1])
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=[lo_off, hi_off],
                        marker="o",
                        ls=ls,
                        ms=3.5,
                        label=f"{base}|{slot}",
                        lw=1.1,
                        capsize=2,
                    )
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel(xlab + " (0 = uncrossed base cell)")
        ax.set_title(f"{prefix} curves (steered F_beh)")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("steered F_beh")
    _save(fig, out_dir, "dose_position", inputs)


def fig_margin_validation(
    margin_cells: list[dict],
    f_cells: list[dict],
    validation: dict,
    out_dir: Path,
    inputs: list[Path],
) -> None:
    f_by_key = {(r["pair_id"], r["slot"]): r["f_beh"] for r in f_cells if r["f_beh"] is not None}
    xs, ys = [], []
    for r in margin_cells:
        if r["arm"] != "steered" or r.get("margin_shift") is None:
            continue
        f = f_by_key.get((r["pair_id"], r["slot"]))
        if f is not None:
            xs.append(r["margin_shift"])
            ys.append(f)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xs, ys, s=8, alpha=0.5, color="#4878d0")
    rho = validation.get("rho_margin_fbeh")
    ax.set_xlabel("TF fixed-pool margin shift (patched - floor anchor)")
    ax.set_ylabel("per-pair F_beh (steered)")
    ax.set_title(
        f"Margin validation (rule 19): Spearman rho={rho if rho is None else round(rho, 3)} "
        f"(n={validation.get('n')}, validated={validation.get('validated')})"
    )
    _save(fig, out_dir, "margin_validation", inputs)


def fig_diagnostics(stats: dict, f_cells: list[dict], out_dir: Path, inputs: list[Path]) -> None:
    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"coh": 0, "cap": 0, "n": 0})
    for r in f_cells:
        k = f"{r['cell']}|{r['slot']}"
        agg[k]["coh"] += r["n_coherent"]
        agg[k]["cap"] += r["n_cap_hit"]
        agg[k]["n"] += r["n_draws"]
    items = _cells_sorted(stats["per_cell"])
    labels = [f"{r['cell']}|{r['slot']}" for _, r in items]
    coh = [agg[k]["coh"] / max(agg[k]["n"], 1) for k in labels]
    cap = [agg[k]["cap"] / max(agg[k]["n"], 1) for k in labels]
    n_post = [r["n_post_exclusion"] for _, r in items]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(labels) * 0.3), 9), sharex=True)
    axes[0].bar(x, coh, color="#4878d0")
    axes[0].set_ylabel("coherent fraction")
    axes[0].axhline(0.9, color="k", lw=0.6, ls=":")
    axes[1].bar(x, cap, color="#ee854a")
    axes[1].set_ylabel("cap-hit fraction")
    axes[1].axhline(0.02, color="k", lw=0.6, ls=":")
    axes[2].bar(x, n_post, color="#6acc64")
    axes[2].axhline(12, color="k", lw=0.6, ls=":")
    axes[2].set_ylabel("post-exclusion n (pairs)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=90, fontsize=5.5)
    fig.suptitle("Diagnostics: coherence, cap-hit (2% line), separation survival (floor 12)")
    _save(fig, out_dir, "diagnostics", inputs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2162 figure set.")
    ap.add_argument("--metrics-dir", type=Path, default=Path("eval_results/issue_2162/f_metrics"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/issue_2162"))
    ap.add_argument("--only", default=None, help="comma-separated figure subset (manifest keys)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    md = args.metrics_dir
    stats = json.loads((md / "stats.json").read_text())
    f_cells = list(_iter_jsonl(md / "f_cells.jsonl"))
    manifest = {
        "hero": lambda: fig_hero(stats, f_cells, args.out_dir, [md / "stats.json"]),
        "two_by_two": lambda: fig_two_by_two(
            json.loads((md / "two_by_two.json").read_text()),
            args.out_dir,
            [md / "two_by_two.json"],
        ),
        "layer_profile": lambda: fig_layer_profile(
            json.loads((md / "probe.json").read_text()), args.out_dir, [md / "probe.json"]
        ),
        "route_contrasts": lambda: fig_route_contrasts(stats, args.out_dir, [md / "stats.json"]),
        "dose_position": lambda: fig_dose_position(stats, args.out_dir, [md / "stats.json"]),
        "margin_validation": lambda: fig_margin_validation(
            list(_iter_jsonl(md / "margin_cells.jsonl")),
            f_cells,
            json.loads((md / "margin_validation.json").read_text()),
            args.out_dir,
            [md / "margin_cells.jsonl", md / "margin_validation.json"],
        ),
        "diagnostics": lambda: fig_diagnostics(
            stats, f_cells, args.out_dir, [md / "f_cells.jsonl"]
        ),
    }
    only = set(args.only.split(",")) if args.only else set(manifest)
    unknown = only - set(manifest)
    assert not unknown, f"unknown figure keys: {sorted(unknown)}"
    for name in manifest:
        if name in only:
            logger.info("[figures] building %s", name)
            manifest[name]()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
