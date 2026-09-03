#!/usr/bin/env python3
"""Render the 28x28 cross-layer ridge grid appendix figure (c2a-v2).

Two heatmap panels over (context layer, answer layer) cells of the cross-layer
ridge grid: held-out R^2 and top-1 retrieval (whitened cosine + CSLS). Style,
canvas scale, fonts, and export come from ``c2a_plot_style``; the exported PDF
is asserted to be full text width (942.857 pt) with only Inter fonts embedded.
The optional bootstrap contrast file is printed to stdout, never drawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# #847: thread caps must land BEFORE the matplotlib/numpy imports below. On the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and
# the BLAS pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    INK,
    PAPER,
    ROLES,
    STYLE_VERSION,
    c2a_figure,
    canvas_width_in,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)

DEFAULT_GRID = ROOT / "eval_results/issue_1901/xlayer_grid/grid.json"
SMOKE_GRID = Path("/mnt/eps-data/thomasjiralerspong/issue1901_xlayer_smoke/out_eval/grid.json")
DEFAULT_OUT_STEM = ROOT / "figures/paper/c1_crosslayer_grid"


def _cell_matrix(grid: dict, extract) -> np.ndarray:
    """Dense (context layer, answer layer) matrix; NaN where a cell is absent."""

    layers_c, layers_a = grid["layers_c"], grid["layers_a"]
    out = np.full((len(layers_c), len(layers_a)), np.nan)
    for i, lc in enumerate(layers_c):
        row = grid["cells"].get(str(lc), {})
        for j, la in enumerate(layers_a):
            cell = row.get(str(la))
            if cell is not None:
                out[i, j] = float(extract(cell))
    return out


def _load_grid(path: Path) -> dict:
    source = json.loads(path.read_text())
    layers_c = [int(v) for v in source["layers_c"]]
    layers_a = [int(v) for v in source["layers_a"]]
    assert layers_c == sorted(layers_c) and layers_a == sorted(layers_a)
    grid = {"layers_c": layers_c, "layers_a": layers_a, "cells": source["cells"]}
    r2 = _cell_matrix(grid, lambda c: c["ridge"]["whole_map_r2"])
    top1 = _cell_matrix(grid, lambda c: c["ridge"]["retrieval"]["whiten_csls"]["acc_at_k"]["1"])
    identity_r2 = _cell_matrix(grid, lambda c: c["identity_bias"]["whole_map_r2"])
    lam = _cell_matrix(grid, lambda c: c["fit_meta"]["selected_lambda"])
    present = ~np.isnan(r2)
    assert present.any(), f"no cells present in {path}"
    assert np.all(r2[present] <= 1.0 + 1e-9)
    assert np.all((top1[~np.isnan(top1)] >= 0.0) & (top1[~np.isnan(top1)] <= 1.0))
    any_cell = next(iter(next(iter(source["cells"].values())).values()))
    return {
        "layers_c": layers_c,
        "layers_a": layers_a,
        "r2": r2,
        "top1": top1,
        "identity_r2": identity_r2,
        "selected_lambda": lam,
        "n_train": int(any_cell["n_train"]),
        "chance_at_1": float(any_cell["ridge"]["chance_at_1"]),
        "n_cells_present": int(present.sum()),
    }


def _layer_ticks(layers: list[int]) -> tuple[list[int], list[str]]:
    """Every 4th layer labeled (0, 4, ..., 24) plus the last layer; small subsets in full."""

    if len(layers) <= 10:
        return list(range(len(layers))), [str(v) for v in layers]
    positions = [i for i, v in enumerate(layers) if v % 4 == 0 or i == len(layers) - 1]
    return positions, [str(layers[i]) for i in positions]


def _heat_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    grid: dict,
    matrix: np.ndarray,
    *,
    letter: str,
    kicker: str,
    title: str,
    vmin: float,
) -> dict:
    layers_c, layers_a = grid["layers_c"], grid["layers_a"]
    cmap = LinearSegmentedColormap.from_list("c2a_seq_teal", [PAPER, ROLES["linear"].color])
    cmap.set_bad(GRID)
    masked = np.ma.masked_invalid(matrix)
    vmax = float(np.nanmax(matrix))
    im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    # Diagonal (equal-layer) cells: thin outlines. Grid maximum: heavy outline.
    for i, lc in enumerate(layers_c):
        for j, la in enumerate(layers_a):
            if lc == la and not np.isnan(matrix[i, j]):
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.9, alpha=0.7
                    )
                )
    i_max, j_max = np.unravel_index(np.nanargmax(matrix), matrix.shape)
    ax.add_patch(Rectangle((j_max - 0.5, i_max - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2.4))

    style_axis(ax, grid_axis="none")
    x_pos, x_lab = _layer_ticks(layers_a)
    y_pos, y_lab = _layer_ticks(layers_c)
    ax.set_xticks(x_pos, x_lab)
    ax.set_yticks(y_pos, y_lab)
    ax.set_xlabel("Answer layer")
    ax.set_ylabel("Context layer")
    panel_header(ax, letter, kicker, title)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=6)
    return {
        "max_cell": {"layer_c": layers_c[i_max], "layer_a": layers_a[j_max]},
        "max_value": float(matrix[i_max, j_max]),
        "value_range": [float(np.nanmin(matrix)), vmax],
        "color_vmin": vmin,
    }


def make_figure(grid: dict) -> tuple[plt.Figure, float, dict]:
    set_c2a_style()
    fig, include_frac = c2a_figure("full", aspect=0.46)
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.96, top=0.78, bottom=0.11, wspace=0.30)
    ax_r2 = fig.add_subplot(gs[0, 0])
    ax_top1 = fig.add_subplot(gs[0, 1])

    r2_min = float(np.nanmin(grid["r2"]))
    panel_a = _heat_panel(
        fig,
        ax_r2,
        grid,
        grid["r2"],
        letter="A",
        kicker=f"{grid['n_train']:,} training contexts",
        title="Held-out $R^2$ of the ridge map",
        vmin=0.0,
    )
    if r2_min < 0.0:
        print(f"note: {int((grid['r2'] < 0).sum())} cells have R^2 < 0; color scale floors at 0")
    panel_b = _heat_panel(
        fig,
        ax_top1,
        grid,
        grid["top1"],
        letter="B",
        kicker=f"chance {grid['chance_at_1']:.1%}",
        title="Top-1 retrieval of the ridge map",
        vmin=0.0,
    )
    return fig, include_frac, {"r2": panel_a, "top1": panel_b}


def _print_bootstrap_headline(path: Path) -> None:
    contrasts = json.loads(path.read_text())["contrasts"]
    for name in ("r2", "acc1_wcsls"):
        c = contrasts[name]
        best, off = c["best_diag_cell"], c["best_offdiag_cell"]
        fixed, aware = c["fixed_cells_ci"], c["selection_aware_ci"]
        print(
            f"bootstrap {name}: best diag c{best['layer_c']}:a{best['layer_a']} vs "
            f"best offdiag c{off['layer_c']}:a{off['layer_a']}; "
            f"diff {c['point_diff']:+.4f} "
            f"(fixed 95% CI [{fixed['lo']:+.4f}, {fixed['hi']:+.4f}]; "
            f"selection-aware [{aware['lo']:+.4f}, {aware['hi']:+.4f}])"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, str | bool | None]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "tracked_worktree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _assert_pdf_width(pdf: Path, expected_pt: float, tol_pt: float = 0.5) -> float:
    if shutil.which("pdfinfo") is None:
        raise RuntimeError("pdfinfo not found; cannot verify the exported PDF width")
    info = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True)
    match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", info.stdout)
    if match is None:
        raise RuntimeError(f"could not parse page size from pdfinfo output for {pdf}")
    width = float(match.group(1))
    if abs(width - expected_pt) > tol_pt:
        raise AssertionError(f"{pdf} is {width} pt wide; expected {expected_pt:.3f} +/- {tol_pt}")
    return width


def _assert_inter_only(pdf: Path) -> list[str]:
    if shutil.which("pdffonts") is None:
        raise RuntimeError("pdffonts not found; cannot verify the embedded fonts")
    out = subprocess.run(["pdffonts", str(pdf)], check=True, capture_output=True, text=True)
    lines = out.stdout.splitlines()
    fonts = [line.split()[0] for line in lines[2:] if line.strip()]
    if not fonts:
        raise AssertionError(f"{pdf} embeds no fonts at all")
    non_inter = [name for name in fonts if "Inter" not in name]
    if non_inter:
        raise AssertionError(f"{pdf} embeds non-Inter fonts: {non_inter}")
    return fonts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--grid", type=Path, default=None, help=f"default {DEFAULT_GRID}")
    parser.add_argument(
        "--bootstrap", type=Path, default=None, help="default: sibling bootstrap.json, optional"
    )
    parser.add_argument("--out-stem", type=Path, default=DEFAULT_OUT_STEM)
    parser.add_argument(
        "--smoke", action="store_true", help=f"render the 2x2 smoke grid at {SMOKE_GRID}"
    )
    args = parser.parse_args()

    grid_path = args.grid if args.grid is not None else (SMOKE_GRID if args.smoke else DEFAULT_GRID)
    out_stem: Path = args.out_stem
    if args.smoke and out_stem.resolve() == DEFAULT_OUT_STEM.resolve():
        raise SystemExit(
            "--smoke refuses the canonical figures/paper stem; pass --out-stem <scratch path>"
        )
    bootstrap_path = args.bootstrap
    if bootstrap_path is None:
        sibling = grid_path.parent / "bootstrap.json"
        bootstrap_path = sibling if sibling.exists() else None
        if bootstrap_path is None:
            print(f"note: no bootstrap.json next to {grid_path}; skipping the headline contrast")
    grid = _load_grid(grid_path)
    print(
        f"grid: {len(grid['layers_c'])}x{len(grid['layers_a'])} layers, "
        f"{grid['n_cells_present']} cells present"
    )

    git_state = _git_state()
    fig, include_frac, panels = make_figure(grid)
    outputs = save_c2a_figure(
        fig,
        out_stem,
        title="Cross-layer ridge grid",
        subject="Held-out R^2 and top-1 retrieval over (context layer, answer layer) cells",
        creator="scripts/paper_fig_crosslayer_grid.py",
        include_width=include_frac,
    )
    plt.close(fig)

    metadata = out_stem.with_suffix(".meta.json")
    metadata.write_text(
        json.dumps(
            {
                "status": "cross-layer ridge grid appendix figure",
                "style_version": STYLE_VERSION,
                "plotting_script": "scripts/paper_fig_crosslayer_grid.py",
                "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
                "reproduction_command": "uv run python scripts/paper_fig_crosslayer_grid.py",
                "git": git_state,
                "sources": {
                    "grid": {"path": str(grid_path), "sha256": _sha256(grid_path)},
                    "bootstrap": (
                        {"path": str(bootstrap_path), "sha256": _sha256(bootstrap_path)}
                        if bootstrap_path is not None
                        else None
                    ),
                },
                "layers_c": grid["layers_c"],
                "layers_a": grid["layers_a"],
                "panels": panels,
                "plotted": {
                    "r2": np.where(np.isnan(grid["r2"]), None, grid["r2"].round(6)).tolist(),
                    "top1": np.where(np.isnan(grid["top1"]), None, grid["top1"].round(6)).tolist(),
                    "identity_bias_r2": np.where(
                        np.isnan(grid["identity_r2"]), None, grid["identity_r2"].round(6)
                    ).tolist(),
                    "selected_lambda": np.where(
                        np.isnan(grid["selected_lambda"]), None, grid["selected_lambda"]
                    ).tolist(),
                },
                "save_record": outputs["record"],
                "output_sha256": {
                    kind: _sha256(path) for kind, path in outputs.items() if isinstance(path, Path)
                },
            },
            indent=2,
        )
        + "\n"
    )

    width_pt = _assert_pdf_width(outputs["pdf"], expected_pt=canvas_width_in(include_frac) * 72.0)
    fonts = _assert_inter_only(outputs["pdf"])
    print(f"pdf width: {width_pt} pt (expected {canvas_width_in(include_frac) * 72.0:.3f})")
    print(f"embedded fonts: {fonts}")
    if bootstrap_path is not None:
        _print_bootstrap_headline(bootstrap_path)
    for kind, path in {**outputs, "metadata": metadata}.items():
        if isinstance(path, Path):
            print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
