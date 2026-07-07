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

from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style  # noqa: E402
from explore_persona_space.experiments import issue_1112 as C  # noqa: E402

logger = logging.getLogger("issue1112.figures")

SYCO_CELLS = ("s1_lora_neg", "s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos")
CELL_LABEL = {
    "s1_lora_neg": "LoRA + negatives (reused)",
    "s2_lora_pos": "LoRA positives-only",
    "s3_fullft_neg": "Full-FT + negatives",
    "s4_fullft_pos": "Full-FT positives-only",
    "s5_lora_generic": "Generic control (LoRA)",
    "s6_fullft_generic": "Generic control (full-FT)",
    "m1_lora_band8": "Marker LoRA (~8 nat)",
    "m2_fullft_band8": "Marker full-FT (~8 nat)",
}
DVS = ("rank_k_at_90", "pr_lambda", "top_share_lambda", "mu_norm")


def _style_for(cell: str, palette: list[str]) -> dict:
    """color = method, linestyle = negatives (plan §6); generics grey."""
    if cell in C.GENERIC_CELLS:
        return {"color": "0.6", "linestyle": "-" if "lora" in cell else "--", "alpha": 0.8}
    color = palette[0] if "lora" in cell else palette[1]
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
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    logger.info("[figures] wrote %s", png)


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
                for cell in present:
                    layers = sorted(by_cell[cell])
                    ax.plot(
                        layers,
                        [by_cell[cell][li][dv] for li in layers],
                        label=_install_label(cell, installs),
                        **_style_for(cell, palette),
                    )
                    lo_hi = [by_cell[cell][li]["boot_ci"].get(dv) for li in layers]
                    if all(x is not None for x in lo_hi):
                        ax.fill_between(
                            layers,
                            [x[0] for x in lo_hi],
                            [x[1] for x in lo_hi],
                            alpha=0.12,
                            color=_style_for(cell, palette)["color"],
                        )
                if dv == "rank_k_at_90" and arm == "response" and group == "syco":
                    name = "hero_syco_rankk_profiles"
                else:
                    name = f"profile_{group}_{arm}_{dv}"
                ax.set_xlabel("decoder layer")
                ax.set_ylabel(dv)
                ax.set_title(f"{group} Δx {dv} — {arm} arm (selected dose)")
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
    for i, cell in enumerate(present):
        rec = by_cell[cell][C.PRIMARY_LAYER]
        v = rec["rank_k_at_90"]
        lo, hi = rec["boot_ci"]["rank_k_at_90"]
        style = _style_for(cell, palette)
        ax.bar(i, v, color=style["color"], hatch="" if style["linestyle"] == "-" else "//")
        ax.errorbar(i, v, yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]], color="k", capsize=3)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([CELL_LABEL[c] for c in present], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("rank-k@90")
    ax.set_title(
        f"2×2 rank-k@90 at layer {C.PRIMARY_LAYER} (response arm, 95% cluster-bootstrap CI)"
    )
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
        ax.set_xlabel("decoder layer")
        ax.set_ylabel(dv)
        ax.set_title(f"{dv} vs r_B — response arm")
        ax.legend(fontsize=7)
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
    ax.set_xticklabels(order)
    ax.set_ylabel("rank-k@90")
    ax.set_title(f"dose stability — layer {C.PRIMARY_LAYER}, response arm")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "explore_dose_stability_rankk", _meta(cells=sorted(multi)))


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
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for x, y, cell in pts:
        ax.scatter(x, y)
        ax.annotate(
            CELL_LABEL.get(cell, cell),
            (x, y),
            fontsize=6,
            textcoords="offset points",
            xytext=(4, 3),
        )
    ax.set_xlabel("Tier-1 judged rate (selected rung)")
    ax.set_ylabel(f"rank-k@90 @ L{C.PRIMARY_LAYER}")
    ax.set_title("geometry DV vs install rate")
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
    ax.set_ylabel("rank-k@90")
    ax.set_title("marker Δx rank-k@90 — LoRA vs full-FT (response arm)")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "hero_marker_rankk_profiles", _meta(cells=present, delta_g=delta_g))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="#1112 figures (plan §6).")
    p.add_argument("--geometry-json", type=Path, required=True)
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
    args = p.parse_args(argv)

    set_paper_style()
    payload = json.loads(args.geometry_json.read_text())
    records = payload["records"]
    installs = install_ladders_fig(args.selection_dir, args.out_dir)
    profile_figs(records, installs, args.out_dir)
    layer14_bar(records, args.out_dir)
    cos_rb_figs(records, args.out_dir)
    dose_stability_fig(records, args.out_dir)
    dv_vs_install_fig(records, installs, args.out_dir)
    marker_fig(records, args.marker_dir, args.out_dir)
    logger.info("[figures] done -> %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
