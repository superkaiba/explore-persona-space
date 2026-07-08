#!/usr/bin/env python
# ruff: noqa: RUF001, RUF002
"""#1112 figures driver (VM, CPU; plan §6 "Figures").

Consumes ``geometry_per_cell.json`` (+ optionally the selection/ladder JSONs
and marker slot reads) and renders the plan-§6 set: hero per-layer rank-k@90 /
PR profiles (4 sycophancy cells, color = method, linestyle = negatives,
generic controls grey, install rates in the legend), the layer-14 2×2 bar with
cluster-bootstrap CIs, the marker LoRA-vs-FT profile with realized ΔG, and the
exploratory dump (per-arm grids, top-share / ‖μ‖ profiles, cos-to-r_B profiles
vs the norm-matched random band, dose-stability trajectories, install
ladders). Missing optional inputs are logged and skipped (the analyzer re-runs
with the full inputs); every produced figure lands under ``--out-dir``.

Smoke (same code path):
    uv run python scripts/issue1112_figures.py \
        --geometry-json <scratch>/geometry_per_cell.json \
        --out-dir /tmp/issue-1112-smoke/figures
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments import issue_1112 as C  # noqa: E402

logger = logging.getLogger("issue1112.figures")

SYCO_CELLS = ("s1_lora_neg", "s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos")
CELL_LABEL = {
    "s1_lora_neg": "LoRA + negatives",
    "s2_lora_pos": "LoRA positives-only",
    "s3_fullft_neg": "Full-FT + negatives",
    "s4_fullft_pos": "Full-FT positives-only",
    "s5_lora_generic": "Generic control (LoRA)",
    "s6_fullft_generic": "Generic control (full-FT)",
    "m1_lora_band8": "Marker LoRA",
    "m2_fullft_band8": "Marker full-FT",
    "s5_lora_neg_lr5e6": "lr-matched LoRA + negatives",
}
# Realized learning rate per sycophancy cell (plan v8 hero legend: lr + rate).
CELL_LR_TEXT = {
    "s1_lora_neg": "lr 1e-5",
    "s2_lora_pos": "lr 1e-5",
    "s3_fullft_neg": "lr 5e-6",
    "s4_fullft_pos": "lr 5e-6",
    "s5_lora_generic": "lr 1e-5",
    "s6_fullft_generic": "lr 5e-6",
    "s5_lora_neg_lr5e6": "lr 5e-6",
}
DVS = ("rank_k_at_90", "pr_lambda", "top_share_lambda", "mu_norm")
DV_LABEL = {
    "rank_k_at_90": "rank-k@90 (modes for 90% of variance)",
    "pr_lambda": "participation ratio of the shift spectrum",
    "top_share_lambda": "top-eigenvalue share of shift variance",
    "mu_norm": "mean-shift norm (residual-stream units)",
}


def _style_for(cell: str, palette: list[str]) -> dict:
    """color = method, linestyle = negatives (plan §6); generics grey; marker cells solid."""
    if cell == C.LR_MATCHED_CELL:
        # LoRA color (color = method) at a distinct dash-dot: the round's cell.
        return {"color": palette[0], "linestyle": "-.", "linewidth": 2.0}
    if cell in C.GENERIC_CELLS:
        return {"color": "0.6", "linestyle": "-" if "lora" in cell else "--", "alpha": 0.8}
    color = palette[0] if "lora" in cell else palette[1]
    if cell in C.MARKER_CELLS:
        return {"color": color, "linestyle": "-"}
    linestyle = "-" if cell.endswith("_neg") or cell == "s1_lora_neg" else "--"
    return {"color": color, "linestyle": linestyle}


def _records_by_cell(records: dict, *, arm: str, dose_by_cell: dict[str, str]) -> dict:
    """cell -> {layer: record} at the cell's read dose for one arm."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for _key, rec in records.items():
        if rec["arm"] != arm:
            continue
        if rec["dose"] != dose_by_cell.get(rec["cell"], "selected"):
            continue
        out[rec["cell"]][int(rec["layer"])] = rec
    return out


def _save(fig, out_dir: Path, name: str, meta: dict) -> None:
    """Save PNG+PDF+meta via ``savefig_paper``; merge run-specific meta keys in."""
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, name, dir=out_dir)
    plt.close(fig)
    meta_path = out_dir / f"{name}.meta.json"
    sidecar = json.loads(meta_path.read_text())
    sidecar.update({k: v for k, v in meta.items() if k not in sidecar})
    meta_path.write_text(json.dumps(sidecar, indent=1) + "\n")
    logger.info("[figures] wrote %s", out_dir / f"{name}.png")


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _meta(**kw) -> dict:
    return {
        "git_commit": _git_commit(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **kw,
    }


def _install_label(cell: str, installs: dict) -> str:
    label = CELL_LABEL.get(cell, cell)
    rate = installs.get(cell)
    return f"{label} (rate {rate:.2f})" if isinstance(rate, int | float) else label


def profile_figs(records: dict, installs: dict, out_dir: Path) -> None:
    palette = paper_palette(4)
    for arm in C.CAPTURE_ARMS:
        by_cell = _records_by_cell(records, arm=arm, dose_by_cell={})
        for group, cells in (("syco", (*SYCO_CELLS, *C.GENERIC_CELLS)), ("marker", C.MARKER_CELLS)):
            present = [c for c in cells if c in by_cell]
            if not present:
                continue
            for dv in DVS:
                fig, ax = plt.subplots(figsize=(6.4, 4.0))
                # NOTE: per-cell bootstrap bands deliberately DROPPED — a
                # with-replacement cluster resample keeps ~63% unique rows, so
                # the per-cell interval reflects a smaller effective n, not a
                # CI for the full-cloud point; paired-difference CIs (text)
                # are the inferential objects.
                for cell in present:
                    layers = sorted(by_cell[cell])
                    ax.plot(
                        layers,
                        [by_cell[cell][li][dv] for li in layers],
                        label=_install_label(cell, installs),
                        **_style_for(cell, palette),
                    )
                if dv == "rank_k_at_90" and arm == "response" and group == "syco":
                    name = "hero_syco_rankk_profiles"
                else:
                    name = f"profile_{group}_{arm}_{dv}"
                group_word = "sycophancy" if group == "syco" else "marker"
                if dv == "mu_norm":
                    ax.set_yscale(
                        "log"
                    )  # late layers dominate linearly; log keeps mid-layers legible
                ax.set_xlabel("decoder layer")
                ax.set_ylabel(DV_LABEL[dv])
                ax.set_title(f"{group_word} activation shift — {arm} arm (selected checkpoint)")
                ax.legend(fontsize=7)
                _save(fig, out_dir, name, _meta(arm=arm, dv=dv, cells=present))


def layer14_bar(records: dict, out_dir: Path) -> None:
    palette = paper_palette(4)
    by_cell = _records_by_cell(records, arm="response", dose_by_cell={})
    present = [c for c in SYCO_CELLS if c in by_cell and C.PRIMARY_LAYER in by_cell[c]]
    if len(present) < 2:
        logger.info("[figures] skip layer14 bar (only %s present)", present)
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    # Per-cell bootstrap whiskers DROPPED (deflated resample spread, not a CI
    # for the 120-row point); paired-difference CIs are reported in the text.
    for i, cell in enumerate(present):
        rec = by_cell[cell][C.PRIMARY_LAYER]
        v = rec["rank_k_at_90"]
        style = _style_for(cell, palette)
        bars = ax.bar(i, v, color=style["color"], hatch="" if style["linestyle"] == "-" else "//")
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([CELL_LABEL[c] for c in present], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("rank-k@90 (modes for 90% of variance)")
    ax.set_title(f"2×2 rank-k@90 at layer {C.PRIMARY_LAYER} (response arm)")
    _save(fig, out_dir, "hero_syco_2x2_layer14", _meta(cells=present))


def cos_rb_figs(records: dict, out_dir: Path) -> None:
    palette = paper_palette(4)
    for dv in ("cos_top_to_rb", "cos_mu_to_rb"):
        by_cell = _records_by_cell(records, arm="response", dose_by_cell={})
        present = [c for c in (*SYCO_CELLS, *C.GENERIC_CELLS, *C.MARKER_CELLS) if c in by_cell]
        present = [c for c in present if any(dv in r for r in by_cell[c].values())]
        if not present:
            logger.info("[figures] skip %s (no records carry it)", dv)
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        band_done = False
        for cell in present:
            layers = sorted(li for li in by_cell[cell] if dv in by_cell[cell][li])
            ax.plot(
                layers,
                [by_cell[cell][li][dv] for li in layers],
                label=CELL_LABEL.get(cell, cell),
                **_style_for(cell, palette),
            )
            if not band_done:
                cis = [by_cell[cell][li].get("random_cos_ci") for li in layers]
                if all(isinstance(x, dict) for x in cis):
                    ax.fill_between(
                        layers,
                        [x["ci_low"] for x in cis],
                        [x["ci_high"] for x in cis],
                        color="0.8",
                        alpha=0.5,
                        label="norm-matched random cos CI",
                    )
                    band_done = True
        which = "top shift direction" if dv == "cos_top_to_rb" else "mean shift direction"
        short = "top direction" if dv == "cos_top_to_rb" else "mean shift"
        ax.set_xlabel("decoder layer")
        ax.set_ylabel(f"cos({short}, behavior read-out)")
        ax.set_title(f"alignment of the {which} to the behavior read-out — response arm")
        ax.legend(fontsize=6, ncol=2)
        _save(fig, out_dir, f"explore_{dv}_profiles", _meta(dv=dv, cells=present))


def dose_stability_fig(records: dict, out_dir: Path) -> None:
    palette = paper_palette(4)
    per_cell: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in records.values():
        if rec["arm"] == "response" and int(rec["layer"]) == C.PRIMARY_LAYER:
            per_cell[rec["cell"]][rec["dose"]] = rec["rank_k_at_90"]
    multi = {c: d for c, d in per_cell.items() if len(d) > 1}
    if not multi:
        logger.info("[figures] skip dose stability (no multi-dose captures)")
        return
    order = ["step6", "selected", "step30"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for cell, doses in sorted(multi.items()):
        xs = [order.index(d) for d in order if d in doses]
        ax.plot(
            xs,
            [doses[order[x]] for x in xs],
            marker="o",
            label=CELL_LABEL.get(cell, cell),
            **_style_for(cell, palette),
        )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["step 6", "selected checkpoint", "step 30"])
    ax.set_ylabel("rank-k@90 (modes for 90% of variance)")
    ax.set_title(f"rank-k@90 across training doses — layer {C.PRIMARY_LAYER}, response arm")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "explore_dose_stability_rankk", _meta(cells=sorted(multi)))


def _read_installs(selection_dir: Path | None) -> dict[str, float]:
    """cell -> selected-checkpoint Tier-1 rate from <cell>/selection.json."""
    installs: dict[str, float] = {}
    if not selection_dir or not selection_dir.exists():
        return installs
    for sel_path in sorted(selection_dir.glob("*/selection.json")):
        sel = json.loads(sel_path.read_text())
        if isinstance(sel.get("rate"), int | float):
            installs[sel_path.parent.name] = float(sel["rate"])
    return installs


def install_ladders_fig(selection_dir: Path, out_dir: Path) -> dict:
    """Per-rung Tier-1 rates per cell; returns cell -> selected rate for legends."""
    installs: dict[str, float] = {}
    if not selection_dir or not selection_dir.exists():
        logger.info("[figures] skip install ladders (no selection dir)")
        return installs
    palette = paper_palette(4)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    any_ladder = False
    for sel_path in sorted(selection_dir.glob("*/selection.json")):
        cell = sel_path.parent.name
        sel = json.loads(sel_path.read_text())
        if isinstance(sel.get("rate"), int | float):
            installs[cell] = float(sel["rate"])
        rates = sel.get("rates_by_step")
        if not rates:
            continue
        steps = sorted(int(s) for s in rates)
        ax.plot(
            steps,
            [float(rates[str(s)]) for s in steps],
            marker="o",
            label=CELL_LABEL.get(cell, cell),
            **_style_for(cell, palette),
        )
        any_ladder = True
    if any_ladder:
        ax.axhspan(0.60, 0.85, color="0.9", label="install band [0.60, 0.85]")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("Tier-1 judged rate")
        ax.set_title("install ladders (Tier-1 rate per rung)")
        ax.legend(fontsize=7)
        _save(fig, out_dir, "explore_install_ladders", _meta(cells=sorted(installs)))
    else:
        plt.close(fig)
    return installs


def dv_vs_install_fig(records: dict, installs: dict, out_dir: Path) -> None:
    by_cell = _records_by_cell(records, arm="response", dose_by_cell={})
    pts = [
        (installs[c], by_cell[c][C.PRIMARY_LAYER]["rank_k_at_90"], c)
        for c in by_cell
        if c in installs and C.PRIMARY_LAYER in by_cell[c]
    ]
    if len(pts) < 2:
        logger.info("[figures] skip DV-vs-install scatter (%d points)", len(pts))
        return
    palette = paper_palette(4)
    # Leader-line offsets keep the two near-coincident LoRA points legible.
    offsets = {
        "s1_lora_neg": (14, -12),
        "s2_lora_pos": (-52, -22),
        "s3_fullft_neg": (8, -16),
        "s4_fullft_pos": (-60, 10),
    }
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for x, y, cell in pts:
        ax.scatter(x, y, color=_style_for(cell, palette)["color"], zorder=3)
        ax.annotate(
            CELL_LABEL.get(cell, cell),
            (x, y),
            fontsize=6,
            textcoords="offset points",
            xytext=offsets.get(cell, (6, 4)),
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.5"},
        )
    ax.set_xlabel("judged sycophancy rate at the selected checkpoint")
    ax.set_ylabel(f"rank-k@90 at layer {C.PRIMARY_LAYER}")
    ax.set_title("shift rank vs installed behavior rate")
    _save(fig, out_dir, "explore_dv_vs_install", _meta(n_points=len(pts)))


def marker_fig(records: dict, marker_dir: Path | None, out_dir: Path) -> None:
    palette = paper_palette(4)
    by_cell = _records_by_cell(records, arm="response", dose_by_cell={})
    present = [c for c in C.MARKER_CELLS if c in by_cell]
    if not present:
        logger.info("[figures] skip marker hero (no marker captures)")
        return
    delta_g: dict[str, float] = {}
    if marker_dir and marker_dir.exists():
        for cell in present:
            p = marker_dir / f"{cell}_slotstats.json"
            if p.exists():
                rec = json.loads(p.read_text())
                v = rec.get("selected_delta_g", rec.get("delta_logp_mean"))
                if isinstance(v, int | float):
                    delta_g[cell] = float(v)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cell in present:
        layers = sorted(by_cell[cell])
        label = CELL_LABEL.get(cell, cell)
        if cell in delta_g:
            label += f" (ΔG {delta_g[cell]:.1f} nat)"
        ax.plot(
            layers,
            [by_cell[cell][li]["rank_k_at_90"] for li in layers],
            label=label,
            **_style_for(cell, palette),
        )
    ax.axvline(
        C.MARKER_READ_LAYER, color="0.7", linestyle=":", label=f"read layer {C.MARKER_READ_LAYER}"
    )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("rank-k@90 (modes for 90% of variance)")
    ax.set_title("marker shift rank-k@90 — LoRA vs full fine-tune (response arm)")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "hero_marker_rankk_profiles", _meta(cells=present, delta_g=delta_g))


def arm_contrast_fig(records: dict, out_dir: Path) -> None:
    """Grouped bar: response vs same-token context rank-k@90 at the primary layer."""
    palette = paper_palette(4)
    cells = [*SYCO_CELLS, *C.GENERIC_CELLS]
    vals: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in records.values():
        if (
            rec["cell"] in cells
            and int(rec["layer"]) == C.PRIMARY_LAYER
            and rec["dose"] == "selected"
            and rec["arm"] in ("response", "context")
        ):
            vals[rec["cell"]][rec["arm"]] = rec["rank_k_at_90"]
    present = [c for c in cells if set(vals[c]) == {"response", "context"}]
    if len(present) < 2:
        logger.info("[figures] skip arm contrast (cells with both arms: %s)", present)
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    width = 0.38
    xs = list(range(len(present)))
    resp = ax.bar(
        [x - width / 2 for x in xs],
        [vals[c]["response"] for c in present],
        width,
        color=palette[2],
        label="own-response arm",
    )
    ctx = ax.bar(
        [x + width / 2 for x in xs],
        [vals[c]["context"] for c in present],
        width,
        color=palette[3],
        label="same-token context arm",
    )
    ax.bar_label(resp, fontsize=7)
    ax.bar_label(ctx, fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABEL[c] for c in present], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("rank-k@90 (modes for 90% of variance)")
    ax.set_title(f"shift rank by measurement arm — layer {C.PRIMARY_LAYER}")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "hero_arm_contrast_rankk", _meta(cells=present))


def spectrum_fig(capture_root: Path | None, out_dir: Path) -> None:
    """Cumulative eigenvalue-share curves at the primary layer (the raw data
    behind rank-k@90): one curve per sycophancy cell, response arm."""
    if capture_root is None or not capture_root.exists():
        logger.info("[figures] skip spectrum fig (no capture root)")
        return
    import numpy as np

    from explore_persona_space.experiments.issue_653.spectral import svd_of_cloud
    from explore_persona_space.experiments.issue_1112.geometry import delta_cloud, load_store

    base_path = capture_root / "base_sycophancy" / "base" / "pooled.pt"
    if not base_path.exists():
        logger.info("[figures] skip spectrum fig (no base store at %s)", base_path)
        return
    base = load_store(base_path)
    palette = paper_palette(4)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    plotted = []
    for cell in (*SYCO_CELLS, *C.GENERIC_CELLS):
        store_path = capture_root / cell / "selected" / "pooled.pt"
        if not store_path.exists():
            continue
        cloud = delta_cloud(load_store(store_path), base, "response", C.PRIMARY_LAYER)
        sigma = svd_of_cloud(cloud, center_rows=True)
        lam = sigma.astype("float64") ** 2

        lam = np.sort(lam)[::-1]
        cum = np.cumsum(lam) / lam.sum()
        ax.plot(
            range(1, len(cum) + 1),
            cum,
            label=CELL_LABEL.get(cell, cell),
            **_style_for(cell, palette),
        )
        plotted.append(cell)
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(0.9, color="0.75", linestyle=":", label="90% of variance")
    ax.set_xlabel("number of leading eigenvalue modes")
    ax.set_ylabel("cumulative share of shift variance")
    ax.set_title(f"shift eigenvalue spectra — layer {C.PRIMARY_LAYER}, response arm")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "explore_spectrum_cumshare_layer14", _meta(cells=plotted))


# ── tf-shared amendment figures (followup `tf-shared-response-capture`) ──────
# Consume geometry_tf_shared.json (self-contained: shared/own/context DVs,
# paired diff CIs, parity cosines, matched-80, layer-14 spectra).

TF_ARM_LABEL = {
    "own": "own-response arm (parent)",
    "shared": "shared-response arm (teacher-forced, new)",
    "context": "same-token context arm (parent)",
}


def _yerr_from_ci(v: float, ci: list[float] | None) -> list[list[float]] | None:
    if not ci:
        return None
    # clamp float-epsilon negatives (constant-bootstrap yerr trap)
    return [[max(0.0, v - ci[0])], [max(0.0, ci[1] - v)]]


def tf_hero_arm_contrast(tf: dict, installs: dict, out_dir: Path) -> None:
    """Hero (plan v6 §6): per cell, three rank-k@90 bars at the primary layer —
    own / shared (teacher-forced) / context — with cluster-bootstrap CIs and
    the registered 30/60 lattice thresholds as reference lines."""
    palette = paper_palette(4)
    records = tf["records"]
    ctx = tf.get("context_primary_layer", {})
    layer = int(tf.get("primary_layer", C.PRIMARY_LAYER))
    cells = [c for c in (*SYCO_CELLS, *C.GENERIC_CELLS) if f"{c}/L{layer}" in records and c in ctx]
    if len(cells) < 2:
        logger.info("[figures] skip tf hero (cells with all three arms: %s)", cells)
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.26
    xs = list(range(len(cells)))
    arm_vals = {
        "own": [(records[f"{c}/L{layer}"]["own"]) for c in cells],
        "shared": [(records[f"{c}/L{layer}"]["shared"]) for c in cells],
        "context": [ctx[c] for c in cells],
    }
    offsets = {"own": -width, "shared": 0.0, "context": width}
    colors = {"own": palette[2], "shared": palette[0], "context": palette[3]}
    # Per-cell bootstrap whiskers DROPPED (deflated resample spread, not a CI
    # for the 120-row point — same convention as the layer-14 2x2 hero);
    # paired own-minus-shared difference CIs are quoted in the text.
    for arm in ("own", "shared", "context"):
        vals = [d["rank_k_at_90"] for d in arm_vals[arm]]
        bars = ax.bar(
            [x + offsets[arm] for x in xs],
            vals,
            width,
            color=colors[arm],
            label=TF_ARM_LABEL[arm],
        )
        ax.bar_label(bars, fontsize=6, fmt="%.0f")
    lat = tf.get("lattice", {}).get("registered_thresholds", {})
    lo = lat.get("collapse_max", 30.0)
    hi = lat.get("stays_diffuse_min", 60.0)
    ax.axhline(lo, color="0.6", linestyle=":", label=f"lattice threshold {lo:.0f}")
    ax.axhline(hi, color="0.4", linestyle="--", label=f"lattice threshold {hi:.0f}")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_install_label(c, installs) for c in cells], rotation=20, ha="right", fontsize=7
    )
    ax.set_ylabel("rank-k@90 (modes for 90% of variance)")
    ax.set_title(f"shift rank by measurement arm — layer {layer} (shared text held fixed)")
    # Headroom above the tallest bar so the legend never overlaps the bars.
    ax.set_ylim(0, 102)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", framealpha=0.9)
    _save(fig, out_dir, "hero_arm_contrast_rankk_tf", _meta(cells=cells, layer=layer))


def tf_profile_figs(tf: dict, out_dir: Path) -> None:
    """Exploratory: per-cell shared-vs-own per-layer profiles for every DV."""
    palette = paper_palette(4)
    records = tf["records"]
    cells = sorted({r["cell"] for r in records.values()})
    if not cells:
        return
    layers_by_cell = {
        c: sorted(int(r["layer"]) for k, r in records.items() if r["cell"] == c) for c in cells
    }
    for dv in DVS:
        n = len(cells)
        ncol = 3
        nrow = -(-n // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
        for i, cell in enumerate(cells):
            ax = axes[i // ncol][i % ncol]
            layers = layers_by_cell[cell]
            style = _style_for(cell, palette)
            for variant, ls in (("own", "-"), ("shared", "--")):
                ax.plot(
                    layers,
                    [records[f"{cell}/L{li}"][variant][dv] for li in layers],
                    color=style["color"],
                    linestyle=ls,
                    label=TF_ARM_LABEL[variant],
                )
            if dv == "mu_norm":
                ax.set_yscale("log")
            ax.set_title(CELL_LABEL.get(cell, cell), fontsize=8)
            if i == 0:
                ax.legend(fontsize=6)
        for j in range(len(cells), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{DV_LABEL[dv]} — own vs shared text (response arm)", fontsize=9)
        fig.supxlabel("decoder layer", fontsize=8)
        _save(fig, out_dir, f"explore_tf_profiles_{dv}", _meta(dv=dv, cells=cells))


def tf_paired_diff_fig(tf: dict, out_dir: Path) -> None:
    """Exploratory: per-layer PAIRED (own − shared) rank-k@90 difference with
    cluster-bootstrap CI bands (the inferential object)."""
    palette = paper_palette(4)
    records = tf["records"]
    cells = sorted({r["cell"] for r in records.values()})
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cell in cells:
        layers = sorted(int(r["layer"]) for k, r in records.items() if r["cell"] == cell)
        diffs = [records[f"{cell}/L{li}"]["diff_own_minus_shared"]["rank_k_at_90"] for li in layers]
        style = _style_for(cell, palette)
        ax.plot(layers, [d["point"] for d in diffs], label=CELL_LABEL.get(cell, cell), **style)
        ax.fill_between(
            layers,
            [d["ci_low"] for d in diffs],
            [d["ci_high"] for d in diffs],
            color=style["color"],
            alpha=0.12,
        )
    ax.axhline(0.0, color="0.7", linestyle=":")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("rank-k@90 difference (own − shared)")
    ax.set_title("text-identity contribution to shift rank — paired difference per layer")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "explore_tf_paired_diff_rankk", _meta(cells=cells))


def tf_spectrum_fig(tf: dict, out_dir: Path) -> None:
    """Exploratory: cumulative eigenvalue-share curves at the primary layer,
    own (solid) vs shared (dashed), from the persisted singular values."""
    import numpy as np

    sv = tf.get("sv_primary_layer", {})
    if not sv:
        return
    palette = paper_palette(4)
    layer = int(tf.get("primary_layer", C.PRIMARY_LAYER))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for cell, both in sorted(sv.items()):
        style = _style_for(cell, palette)
        for variant, ls in (("own", "-"), ("shared", "--")):
            lam = np.sort(np.asarray(both[variant], dtype="float64") ** 2)[::-1]
            if lam.sum() <= 0:
                continue
            cum = np.cumsum(lam) / lam.sum()
            ax.plot(
                range(1, len(cum) + 1),
                cum,
                color=style["color"],
                linestyle=ls,
                label=f"{CELL_LABEL.get(cell, cell)} ({variant})",
            )
    ax.axhline(0.9, color="0.75", linestyle=":", label="90% of variance")
    ax.set_xlabel("number of leading eigenvalue modes")
    ax.set_ylabel("cumulative share of shift variance")
    ax.set_title(f"shift eigenvalue spectra — layer {layer}, own vs shared text")
    ax.legend(fontsize=5, ncol=2)
    _save(fig, out_dir, "explore_tf_spectrum_cumshare", _meta(cells=sorted(sv), layer=layer))


def tf_matched80_fig(tf: dict, out_dir: Path) -> None:
    """Exploratory: matched-80 subsample rank-k@90 (cross-issue comparability)
    beside the full-cloud shared value."""
    m80 = tf.get("matched80_shared", {})
    records = tf["records"]
    layer = int(tf.get("primary_layer", C.PRIMARY_LAYER))
    cells = [c for c in sorted(m80) if "rank_k_at_90_mean" in m80[c] and f"{c}/L{layer}" in records]
    if not cells:
        logger.info("[figures] skip tf matched-80 (no subsampled reads)")
        return
    palette = paper_palette(4)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    width = 0.38
    xs = list(range(len(cells)))
    full = ax.bar(
        [x - width / 2 for x in xs],
        [records[f"{c}/L{layer}"]["shared"]["rank_k_at_90"] for c in cells],
        width,
        color=palette[0],
        label="shared text, full cloud",
    )
    sub = ax.bar(
        [x + width / 2 for x in xs],
        [m80[c]["rank_k_at_90_mean"] for c in cells],
        width,
        color=palette[1],
        label="shared text, matched-80 mean",
    )
    ax.bar_label(full, fontsize=7, fmt="%.0f")
    ax.bar_label(sub, fontsize=7, fmt="%.1f")
    ax.set_xticks(xs)
    ax.set_xticklabels([CELL_LABEL.get(c, c) for c in cells], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("rank-k@90 (modes for 90% of variance)")
    ax.set_title(f"matched-80 comparability read — layer {layer}, shared text")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "explore_tf_matched80", _meta(cells=cells))


def tf_parity_fig(tf: dict, out_dir: Path) -> None:
    """Exploratory: prefix/context parity-cosine distributions (per-row, primary
    layer) + per-layer minima — the pipeline-validation read (WARN bar 0.999)."""
    parity = tf.get("parity", {})
    if not parity:
        return
    palette = paper_palette(4)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    layer = int(tf.get("primary_layer", C.PRIMARY_LAYER))
    for k, arm in enumerate(("prefix", "context")):
        pooled: list[float] = []
        for cell in sorted(parity):
            rows = parity[cell]["arms"][arm].get("per_row_cos_primary_layer") or []
            pooled.extend(rows)
        if pooled:
            axes[0].hist(pooled, bins=40, alpha=0.6, color=palette[k], label=f"{arm} arm")
        for i, cell in enumerate(sorted(parity)):
            per_layer = parity[cell]["arms"][arm]["per_layer"]
            layers = sorted(int(li) for li in per_layer)
            axes[1].plot(
                layers,
                [per_layer[str(li)]["min"] for li in layers],
                color=palette[k],
                alpha=0.35 + 0.1 * i,
                linestyle="-" if arm == "prefix" else "--",
            )
    bar = next(iter(parity.values()))["warn_bar"]
    axes[0].axvline(bar, color="0.3", linestyle=":", label=f"WARN bar {bar}")
    axes[0].set_xlabel(f"per-row cosine (tf vs parent capture), layer {layer}")
    axes[0].set_ylabel("rows")
    axes[0].legend(fontsize=7)
    axes[1].axhline(bar, color="0.3", linestyle=":")
    axes[1].set_xlabel("decoder layer")
    axes[1].set_ylabel("per-layer min cosine")
    fig.suptitle(
        "prefix/context parity check — same prompt tokens across capture rounds", fontsize=9
    )
    _save(fig, out_dir, "explore_tf_parity_cosines", _meta(cells=sorted(parity)))


def tf_figs(tf_geometry_json: Path, installs: dict, out_dir: Path) -> None:
    tf = json.loads(tf_geometry_json.read_text())
    tf_hero_arm_contrast(tf, installs, out_dir)
    tf_profile_figs(tf, out_dir)
    tf_paired_diff_fig(tf, out_dir)
    tf_spectrum_fig(tf, out_dir)
    tf_matched80_fig(tf, out_dir)
    tf_parity_fig(tf, out_dir)


# ── lr-matched amendment figures (followup `lr-matched-method-pair`) ─────────
# Consume geometry_lr_matched.json (+ the parent geometry_per_cell.json as an
# overlay input) and the committed install/*_tier2.json records for legends.


def _read_tier2_installs(install_dir: Path | None) -> dict[str, float]:
    """cell -> Tier-2 trained judged rate from the committed <cell>_tier2.json."""
    installs: dict[str, float] = {}
    if not install_dir or not install_dir.exists():
        return installs
    for p in sorted(install_dir.glob("*_tier2.json")):
        rec = json.loads(p.read_text())
        rate = rec.get("rates", {}).get("trained")
        if isinstance(rate, int | float):
            installs[rec["cell"]] = float(rate)
    return installs


def _lr_legend_label(cell: str, installs: dict) -> str:
    """Plain-English label incl. lr + judged (Tier-2) rate — plan v8 §6."""
    label = CELL_LABEL.get(cell, cell)
    bits = [b for b in (CELL_LR_TEXT.get(cell),) if b]
    rate = installs.get(cell)
    if isinstance(rate, int | float):
        bits.append(f"rate {rate:.2f}")
    return f"{label} ({', '.join(bits)})" if bits else label


def _lr_overlay_records(parent_records: dict, lr: dict, *, arm: str) -> dict[str, dict]:
    """cell -> {layer: record}: the 6 parent cells from geometry_per_cell.json
    plus the NEW cell's records from geometry_lr_matched.json (the re-derived
    s3 duplicate is NOT merged — the parent copy stays authoritative)."""
    by_cell = _records_by_cell(parent_records, arm=arm, dose_by_cell={})
    lr_by_cell = _records_by_cell(lr["records"], arm=arm, dose_by_cell={})
    if C.LR_MATCHED_CELL in lr_by_cell:
        by_cell[C.LR_MATCHED_CELL] = lr_by_cell[C.LR_MATCHED_CELL]
    return by_cell


def lr_hero_mu_norm(parent_records: dict, lr: dict, installs: dict, out_dir: Path) -> None:
    """Hero (plan v8 §6 fig 1): the parent per-layer ‖μ‖ response profiles (6
    cells) EXTENDED with the lr-matched cell, PLUS a side panel with the
    layer-14 paired (s3 − s5) difference CI against zero and the parent's
    (s3 − s1 @ lr 1e-5) reference recomputed under the same draws."""
    palette = paper_palette(4)
    by_cell = _lr_overlay_records(parent_records, lr, arm="response")
    cells = [c for c in (*SYCO_CELLS, *C.GENERIC_CELLS, C.LR_MATCHED_CELL) if c in by_cell]
    if C.LR_MATCHED_CELL not in cells:
        logger.info("[figures] skip lr hero (no lr-matched records)")
        return
    layer = int(lr.get("primary_layer", C.PRIMARY_LAYER))
    pair = lr["lr_matched_pair"]
    diff_matched = pair["mu_norm_diff_by_layer_s3_minus_s5"].get(str(layer))
    diff_ref = pair["reference_s3_minus_s1_by_layer"].get(str(layer))
    fig, (ax, axd) = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.0),
        gridspec_kw={"width_ratios": [2.4, 1.0]},
        layout="constrained",
    )
    for cell in cells:
        layers = sorted(by_cell[cell])
        ax.plot(
            layers,
            [by_cell[cell][li]["mu_norm"] for li in layers],
            label=_lr_legend_label(cell, installs),
            **_style_for(cell, palette),
        )
    ax.set_yscale("log")  # late layers dominate linearly; log keeps mid-layers legible
    ax.set_xlabel("decoder layer")
    ax.set_ylabel(DV_LABEL["mu_norm"])
    ax.set_title("sycophancy mean-shift norm — response arm", loc="left", fontsize=9)
    ax.legend(fontsize=6)
    pts = [
        ("lr-matched pair\n(both lr 5e-6)", diff_matched, palette[1]),
        ("parent pair\n(5e-6 vs 1e-5)", "REF", "0.5"),
    ]
    xs, labels = [], []
    for i, (label, d, color) in enumerate(pts):
        d = diff_ref if d == "REF" else d
        if not d:
            continue
        axd.errorbar(
            i,
            d["point"],
            yerr=[[max(0.0, d["point"] - d["ci_low"])], [max(0.0, d["ci_high"] - d["point"])]],
            fmt="o",
            color=color,
            capsize=4,
        )
        xs.append(i)
        labels.append(label)
    axd.axhline(0.0, color="0.7", linestyle=":")
    axd.set_xticks(xs)
    axd.set_xticklabels(labels, fontsize=6.5)
    axd.set_xlim(-0.6, 1.6)
    axd.set_ylabel(f"‖μ‖ difference at layer {layer} (full-FT − LoRA)")
    axd.set_title("method effect\n(paired bootstrap CI)", fontsize=8)
    _save(
        fig,
        out_dir,
        "hero_syco_mu_norm_lr_matched",
        _meta(cells=cells, layer=layer, mu_n_boot=lr.get("mu_n_boot")),
    )


def lr_ladder_fig(lr: dict, out_dir: Path) -> None:
    """Plan v8 §6 fig 2: the new cell's per-rung Tier-1 judged rate with the
    [0.60, 0.85] band shaded (the parent ladder-figure convention)."""
    sel = lr.get("install", {}).get("selection")
    if not sel or not sel.get("rates_by_step"):
        logger.info("[figures] skip lr ladder (no staged selection record)")
        return
    palette = paper_palette(4)
    rates = sel["rates_by_step"]
    steps = sorted(int(s) for s in rates)
    band = sel.get("band", [0.60, 0.85])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(
        steps,
        [float(rates[str(s)]) for s in steps],
        marker="o",
        label=_lr_legend_label(C.LR_MATCHED_CELL, {}),
        **_style_for(C.LR_MATCHED_CELL, palette),
    )
    if isinstance(sel.get("step"), int) and str(sel["step"]) in rates:
        ax.scatter(
            [sel["step"]],
            [float(rates[str(sel["step"])])],
            s=90,
            facecolors="none",
            edgecolors="0.2",
            zorder=3,
            label=f"selected checkpoint (step {sel['step']})",
        )
    ax.axhspan(band[0], band[1], color="0.9", label=f"install band [{band[0]}, {band[1]}]")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Tier-1 judged rate")
    ax.set_title("install ladder — lr-matched LoRA with negatives (lr 5e-6)")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "explore_install_ladder_lr_matched", _meta(cell=C.LR_MATCHED_CELL))


def lr_rankk_profiles(parent_records: dict, lr: dict, installs: dict, out_dir: Path) -> None:
    """Plan v8 §6 fig 3: rank-k@90 per-layer profile overlay incl. the
    lr-matched cell (the H1 re-check convention)."""
    palette = paper_palette(4)
    by_cell = _lr_overlay_records(parent_records, lr, arm="response")
    cells = [c for c in (*SYCO_CELLS, *C.GENERIC_CELLS, C.LR_MATCHED_CELL) if c in by_cell]
    if C.LR_MATCHED_CELL not in cells:
        logger.info("[figures] skip lr rankk profiles (no lr-matched records)")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cell in cells:
        layers = sorted(by_cell[cell])
        ax.plot(
            layers,
            [by_cell[cell][li]["rank_k_at_90"] for li in layers],
            label=_lr_legend_label(cell, installs),
            **_style_for(cell, palette),
        )
    ax.set_xlabel("decoder layer")
    ax.set_ylabel(DV_LABEL["rank_k_at_90"])
    ax.set_title("shift rank-k@90 — response arm, lr-matched overlay")
    ax.legend(fontsize=6)
    _save(fig, out_dir, "explore_rankk_profiles_lr_matched", _meta(cells=cells))


def lr_mu_diff_by_layer_fig(lr: dict, out_dir: Path) -> None:
    """Exploratory low-level read behind the hero panel: per-layer paired ‖μ‖
    differences with bootstrap CI bands for the matched pair, the parent
    (lr-confounded) reference, and the within-method lr contrast."""
    palette = paper_palette(4)
    pair = lr["lr_matched_pair"]
    series = (
        ("full-FT − lr-matched LoRA (both lr 5e-6)", "mu_norm_diff_by_layer_s3_minus_s5", 1),
        ("full-FT − LoRA @ lr 1e-5 (parent reference)", "reference_s3_minus_s1_by_layer", 3),
        ("LoRA @ lr 1e-5 − lr-matched LoRA (lr effect)", "exploratory_s1_minus_s5_by_layer", 2),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, key, ci in series:
        diffs = pair.get(key, {})
        layers = sorted(int(li) for li in diffs)
        if not layers:
            continue
        color = palette[ci]
        ax.plot(layers, [diffs[str(li)]["point"] for li in layers], label=label, color=color)
        ax.fill_between(
            layers,
            [diffs[str(li)]["ci_low"] for li in layers],
            [diffs[str(li)]["ci_high"] for li in layers],
            color=color,
            alpha=0.12,
        )
    ax.axhline(0.0, color="0.7", linestyle=":")
    ax.set_xlabel("decoder layer")
    ax.set_ylabel("paired ‖μ‖ difference (residual-stream units)")
    ax.set_title("mean-shift-norm differences per layer — paired cluster bootstrap")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "explore_lr_matched_mu_diff_by_layer", _meta(mu_n_boot=lr.get("mu_n_boot")))


def lr_exploratory_figs(parent: dict, lr: dict, installs: dict, out_dir: Path) -> None:
    """Cheap over-produce (plan v8 §6): PR / top-share / cos-to-r_B profile
    overlays incl. the new cell, layer-14 cumulative-spectrum curves for the
    three pair cells, and the matched-80 comparability bars."""
    palette = paper_palette(4)
    parent_records = parent["records"]
    for dv in ("pr_lambda", "top_share_lambda"):
        by_cell = _lr_overlay_records(parent_records, lr, arm="response")
        cells = [c for c in (*SYCO_CELLS, *C.GENERIC_CELLS, C.LR_MATCHED_CELL) if c in by_cell]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for cell in cells:
            layers = sorted(by_cell[cell])
            ax.plot(
                layers,
                [by_cell[cell][li][dv] for li in layers],
                label=_lr_legend_label(cell, installs),
                **_style_for(cell, palette),
            )
        ax.set_xlabel("decoder layer")
        ax.set_ylabel(DV_LABEL[dv])
        ax.set_title(f"{DV_LABEL[dv]} — response arm, lr-matched overlay")
        ax.legend(fontsize=6)
        _save(fig, out_dir, f"explore_lr_matched_{dv}_profiles", _meta(dv=dv, cells=cells))

    # cos(mean shift, behavior read-out) overlay
    by_cell = _lr_overlay_records(parent_records, lr, arm="response")
    cells = [
        c
        for c in (*SYCO_CELLS, *C.GENERIC_CELLS, C.LR_MATCHED_CELL)
        if c in by_cell and any("cos_mu_to_rb" in r for r in by_cell[c].values())
    ]
    if cells:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for cell in cells:
            layers = sorted(li for li in by_cell[cell] if "cos_mu_to_rb" in by_cell[cell][li])
            ax.plot(
                layers,
                [by_cell[cell][li]["cos_mu_to_rb"] for li in layers],
                label=_lr_legend_label(cell, installs),
                **_style_for(cell, palette),
            )
        ax.set_xlabel("decoder layer")
        ax.set_ylabel("cos(mean shift, behavior read-out)")
        ax.set_title("alignment of the mean shift to the behavior read-out — lr-matched overlay")
        ax.legend(fontsize=6, ncol=2)
        _save(fig, out_dir, "explore_lr_matched_cos_mu_to_rb", _meta(cells=cells))

    # layer-14 cumulative eigenvalue-share curves (three pair cells)
    sv = lr.get("sv_primary_layer", {})
    if sv:
        import numpy as np

        layer = int(lr.get("primary_layer", C.PRIMARY_LAYER))
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        for cell, sigma in sorted(sv.items()):
            lam = np.sort(np.asarray(sigma, dtype="float64") ** 2)[::-1]
            if lam.sum() <= 0:
                continue
            cum = np.cumsum(lam) / lam.sum()
            ax.plot(
                range(1, len(cum) + 1),
                cum,
                label=_lr_legend_label(cell, installs),
                **_style_for(cell, palette),
            )
        ax.axhline(0.9, color="0.75", linestyle=":", label="90% of variance")
        ax.set_xlabel("number of leading eigenvalue modes")
        ax.set_ylabel("cumulative share of shift variance")
        ax.set_title(f"shift eigenvalue spectra — layer {layer}, response arm")
        ax.legend(fontsize=6)
        _save(fig, out_dir, "explore_lr_matched_spectrum_cumshare", _meta(cells=sorted(sv)))

    # matched-80 comparability bars (full cloud vs matched-80 mean)
    m80 = {**parent.get("subsample_sensitivity_80row", {}), **lr.get("matched80", {})}
    by_cell = _lr_overlay_records(parent_records, lr, arm="response")
    layer = int(lr.get("primary_layer", C.PRIMARY_LAYER))
    cells = [
        c
        for c in ("s1_lora_neg", "s3_fullft_neg", C.LR_MATCHED_CELL)
        if c in m80 and "rank_k_at_90_mean" in m80[c] and layer in by_cell.get(c, {})
    ]
    if cells:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        width = 0.38
        xs = list(range(len(cells)))
        full = ax.bar(
            [x - width / 2 for x in xs],
            [by_cell[c][layer]["rank_k_at_90"] for c in cells],
            width,
            color=palette[0],
            label="full 120-row cloud",
        )
        sub = ax.bar(
            [x + width / 2 for x in xs],
            [m80[c]["rank_k_at_90_mean"] for c in cells],
            width,
            color=palette[1],
            label="matched-80 subsample mean",
        )
        ax.bar_label(full, fontsize=7, fmt="%.0f")
        ax.bar_label(sub, fontsize=7, fmt="%.1f")
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [CELL_LABEL.get(c, c) for c in cells], rotation=15, ha="right", fontsize=7
        )
        ax.set_ylabel(DV_LABEL["rank_k_at_90"])
        ax.set_title(f"matched-80 comparability read — layer {layer}, response arm")
        ax.legend(fontsize=7)
        _save(fig, out_dir, "explore_lr_matched_matched80", _meta(cells=cells))


def lr_matched_figs(
    parent_geometry_json: Path, lr_geometry_json: Path, install_dir: Path | None, out_dir: Path
) -> None:
    """Render the plan-v8 lr-matched figure set (hero + ladder + rank overlay
    + exploratory dump). The parent JSON is consumed as overlay data only —
    the committed parent figures are NOT re-rendered here."""
    parent = json.loads(parent_geometry_json.read_text())
    lr = json.loads(lr_geometry_json.read_text())
    installs = _read_tier2_installs(install_dir)
    lr_hero_mu_norm(parent["records"], lr, installs, out_dir)
    lr_ladder_fig(lr, out_dir)
    lr_rankk_profiles(parent["records"], lr, installs, out_dir)
    lr_mu_diff_by_layer_fig(lr, out_dir)
    lr_exploratory_figs(parent, lr, installs, out_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="#1112 figures (plan §6).")
    p.add_argument(
        "--geometry-json",
        type=Path,
        default=None,
        help="parent geometry_per_cell.json (optional when only --tf-geometry-json is given)",
    )
    p.add_argument(
        "--tf-geometry-json",
        type=Path,
        default=None,
        help="geometry_tf_shared.json — renders the tf-shared hero + exploratory dump",
    )
    p.add_argument(
        "--lr-geometry-json",
        type=Path,
        default=None,
        help="geometry_lr_matched.json — renders the lr-matched set ONLY (requires "
        "--geometry-json as overlay data; the parent figure set is NOT re-rendered)",
    )
    p.add_argument(
        "--install-dir",
        type=Path,
        default=None,
        help="[--lr-geometry-json] dir holding <cell>_tier2.json (legend lr + rates)",
    )
    p.add_argument(
        "--selection-dir",
        type=Path,
        default=None,
        help="dir holding <cell>/selection.json (ladders + install rates)",
    )
    p.add_argument(
        "--marker-dir",
        type=Path,
        default=None,
        help="dir holding <cell>_slotstats.json (realized ΔG labels)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(f"figures/issue_{C.ISSUE}"),
        help="figure destination (smokes MUST divert to scratch)",
    )
    p.add_argument(
        "--capture-root",
        type=Path,
        default=None,
        help="local <cell>/<dose>/pooled.pt tree (enables the eigenvalue-spectrum figure)",
    )
    args = p.parse_args(argv)

    if args.geometry_json is None and args.tf_geometry_json is None:
        raise SystemExit("need --geometry-json and/or --tf-geometry-json")
    set_paper_style()
    if args.lr_geometry_json is not None:
        if args.geometry_json is None:
            raise SystemExit("--lr-geometry-json requires --geometry-json (overlay data)")
        lr_matched_figs(args.geometry_json, args.lr_geometry_json, args.install_dir, args.out_dir)
        logger.info("[figures] lr-matched set done -> %s", args.out_dir)
        return 0
    if args.geometry_json is not None:
        payload = json.loads(args.geometry_json.read_text())
        records = payload["records"]
        installs = install_ladders_fig(args.selection_dir, args.out_dir)
        profile_figs(records, installs, args.out_dir)
        layer14_bar(records, args.out_dir)
        cos_rb_figs(records, args.out_dir)
        dose_stability_fig(records, args.out_dir)
        dv_vs_install_fig(records, installs, args.out_dir)
        marker_fig(records, args.marker_dir, args.out_dir)
        arm_contrast_fig(records, args.out_dir)
        spectrum_fig(args.capture_root, args.out_dir)
    if args.tf_geometry_json is not None:
        tf_figs(args.tf_geometry_json, _read_installs(args.selection_dir), args.out_dir)
    logger.info("[figures] done -> %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
