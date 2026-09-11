#!/usr/bin/env python3
"""Render the nearest-neighbor-similarity distribution of retrieval failures.

One panel, two overlaid normalized histograms on a single axis: for every
held-out context, the cosine to the nearest OTHER context in the candidate
pool, split by whether the linear context-to-answer map retrieved that
context's answer at top 1.

The two groups differ in size by almost 10x (90 failures against 852
successes), so each histogram is normalized WITHIN its group: the plotted
quantity is the share of that group falling in the bin, over bin edges shared
by both groups.  Raw counts are recorded in the sidecar and never drawn.

No kernel density estimate is used.  Both distributions are bounded above by a
cosine of 1 and the failures pile up against that bound, where a Gaussian
kernel would place mass past 1 and invent a rounded shoulder the data does not
have.  Histograms over shared edges carry the same comparison without the
edge artifact, and the 90 failures are additionally drawn as a rug of the raw
values so no reader has to trust a smoother on a small sample.

The script is plot-only: it reads one banked per-row array file, verifies the
two groups against their recorded summary statistics, and writes a vector PDF,
a color PNG, a grayscale audit PNG, and a provenance JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    ROLES,
    STYLE_VERSION,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    style_axis,
)


DEFAULT_SOURCE = Path(
    os.environ.get(
        "C2A_NN1_PERROW",
        "/mnt/eps-data/thomasjiralerspong/issue1901_ctxsim/paper_fig_inputs/perrow.npz",
    )
)
DEFAULT_OUT = ROOT / "figures/paper"
DEFAULT_STEM = "c3_nn1_failure_distribution"

# Failures take the paper's linear-map hue, the color c3_failures_and_shifts
# already gives the retrieval failures it highlights; the reference population
# they are read against takes the paper-wide control gray, as the candidate-pool
# background does in that same panel.
FAILURE = ROLES["linear"].color
SUCCESS = ROLES["control"].color

BIN_LO, BIN_HI, BIN_WIDTH = 0.40, 1.00, 0.02

# Recorded group statistics of the banked arrays.  A load that does not
# reproduce these is a different file, and the script refuses to plot it.
EXPECTED: dict[str, dict[str, float]] = {
    "failure": {"n": 90, "min": 0.628655, "median": 0.974556, "max": 0.997931},
    "success": {"n": 852, "min": 0.420538, "median": 0.778286, "max": 0.996646},
}
STAT_TOLERANCE = 1e-4

# Provenance of the `correct` column, verified against the banked evaluation
# summary rather than assumed: its 852/942 top-1 rate is exactly
# panel_b.963444.ridge.top1_10000 in the file named below.
RETRIEVAL_ARM = {
    "map": "linear (ridge), trained on 963,444 contexts",
    "model": "Qwen2.5-7B-Instruct",
    "layer": 19,
    "target": "mean(original answer vector + four fresh on-policy answer vectors)",
    "metric": "whitened cosine + two-sided CSLS (K=10), strict top-1",
    "n_query": 942,
    "n_pool": 10000,
    "top1": 852 / 942,
    "cross_check": (
        "eval_results/issue_1901/fig2_pool10k/fig2_pool10k.json "
        "panel_b.963444.ridge.top1_10000 == 0.9044585987261147 == 852/942"
    ),
}

NN1_DEFINITION = (
    "For each of the 942 held-out query contexts, the maximum RAW (unwhitened) "
    "cosine between its layer-19 context vector and the context vector of any "
    "OTHER context in the 10,000-context candidate pool; the query's own "
    "position in the pool is excluded.  Recomputed from c2a_C.npy and "
    "c2a_qcols.npy in the same directory and matching the banked column to "
    "3.0e-07.  Note the asymmetry: this axis is a raw cosine between CONTEXT "
    "vectors, while the failure/success split comes from whitened-cosine CSLS "
    "retrieval in ANSWER space."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


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


def _group_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q75": float(np.percentile(values, 75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _verify(name: str, stats: dict[str, float]) -> list[str]:
    expected = EXPECTED[name]
    problems = []
    if stats["n"] != expected["n"]:
        problems.append(f"{name}: n is {stats['n']}, recorded {int(expected['n'])}")
    for key in ("min", "median", "max"):
        if abs(stats[key] - expected[key]) > STAT_TOLERANCE:
            problems.append(
                f"{name}: {key} is {stats[key]:.6f}, recorded {expected[key]:.6f} "
                f"(tolerance {STAT_TOLERANCE})"
            )
    return problems


def load_data(source: Path) -> dict:
    """Load the per-row arrays, verify both groups, and bin them."""

    if not source.exists():
        raise SystemExit(
            f"per-row array file not found: {source}\n"
            "Pass --source or set C2A_NN1_PERROW to the banked paper_fig_inputs/perrow.npz."
        )
    with np.load(source) as handle:
        missing = [key for key in ("nn1", "correct") if key not in handle.files]
        if missing:
            raise SystemExit(f"{source} is missing required key(s): {missing}")
        nn1 = np.asarray(handle["nn1"], dtype=np.float64)
        correct = np.asarray(handle["correct"]).astype(bool)
    if nn1.shape != correct.shape:
        raise SystemExit(f"nn1 {nn1.shape} and correct {correct.shape} disagree in shape")
    if not np.isfinite(nn1).all():
        raise SystemExit("nn1 carries non-finite values")

    failures, successes = nn1[~correct], nn1[correct]
    stats = {"failure": _group_stats(failures), "success": _group_stats(successes)}
    problems = _verify("failure", stats["failure"]) + _verify("success", stats["success"])
    if problems:
        raise SystemExit(
            "banked group statistics did not reproduce; refusing to plot:\n  "
            + "\n  ".join(problems)
        )

    lo, hi = float(nn1.min()), float(nn1.max())
    if lo < BIN_LO or hi > BIN_HI:
        raise SystemExit(f"values span [{lo:.4f}, {hi:.4f}], outside the bins [{BIN_LO}, {BIN_HI}]")

    n_bins = int(round((BIN_HI - BIN_LO) / BIN_WIDTH))
    edges = np.linspace(BIN_LO, BIN_HI, n_bins + 1)
    groups = {}
    for name, values in (("failure", failures), ("success", successes)):
        counts, _ = np.histogram(values, bins=edges)
        assert counts.sum() == values.size, (name, counts.sum(), values.size)
        groups[name] = {
            **stats[name],
            "counts": counts.tolist(),
            "share": (counts / values.size).tolist(),
            "share_above_0.95": float((values > 0.95).mean()),
            "share_above_0.99": float((values > 0.99).mean()),
        }

    # Rank-based separation, recorded for the caption and never drawn on the
    # canvas: the chance a randomly drawn failure sits closer to its nearest
    # neighbor than a randomly drawn success, ties counted as half.
    comparison = failures[:, None] - successes[None, :]
    auc = float((comparison > 0).mean() + 0.5 * (comparison == 0).mean())

    return {
        "bin_edges": edges.tolist(),
        "bin_width": BIN_WIDTH,
        "groups": groups,
        "failure_values": sorted(float(v) for v in failures),
        "separation": {
            "auc_failure_over_success": auc,
            "median_difference": stats["failure"]["median"] - stats["success"]["median"],
            "definition": (
                "auc_failure_over_success = P(nn1 of a random failure > nn1 of a random "
                "success) + 0.5 P(tie), over all 90 x 852 pairs"
            ),
        },
        "n_total": int(nn1.size),
    }


def make_figure(data: dict, *, rug: bool = True) -> tuple[plt.Figure, float]:
    fig, include_frac = c2a_figure("wide", aspect=0.52)
    grid = fig.add_gridspec(1, 1, left=0.088, right=0.985, top=0.80, bottom=0.155)
    ax = fig.add_subplot(grid[0, 0])

    edges = np.asarray(data["bin_edges"])
    fail = np.asarray(data["groups"]["failure"]["share"])
    success = np.asarray(data["groups"]["success"]["share"])
    n_fail = data["groups"]["failure"]["n"]
    n_success = data["groups"]["success"]["n"]

    # Reference population: a plain filled block, no outline.
    success_patch = ax.stairs(
        success,
        edges,
        fill=True,
        color=SUCCESS,
        alpha=0.45,
        lw=0,
        zorder=2,
        label=f"Retrieval successes (n = {n_success})",
    )
    # Highlighted group: a light wash plus a heavy outline, so the two groups
    # stay separable in the grayscale audit as well as in color.
    ax.stairs(fail, edges, fill=True, color=FAILURE, alpha=0.16, lw=0, zorder=3)
    fail_patch = ax.stairs(
        fail,
        edges,
        fill=False,
        color=FAILURE,
        lw=2.0,
        zorder=4,
        label=f"Retrieval failures (n = {n_fail})",
    )

    top = float(max(fail.max(), success.max()))
    y_max = top * 1.07
    rug_top, rug_bottom = -0.020 * top, -0.055 * top
    ax.set_ylim(rug_bottom * 1.25 if rug else -0.012 * top, y_max)
    if rug:
        values = np.asarray(data["failure_values"])
        ax.vlines(values, rug_bottom, rug_top, color=FAILURE, lw=0.8, alpha=0.75, zorder=5)

    step = 0.1 if top > 0.22 else 0.05
    ticks = np.arange(0.0, top + step * 0.999, step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}".rstrip("0").rstrip(".") if t else "0" for t in ticks])
    ax.set_xlim(BIN_LO, BIN_HI)
    ax.set_xticks(np.arange(BIN_LO, BIN_HI + 1e-9, 0.1))
    ax.set_xticklabels([f"{t:.1f}" for t in np.arange(BIN_LO, BIN_HI + 1e-9, 0.1)])
    ax.set_xlabel("Cosine to nearest other candidate context")
    ax.set_ylabel("Share of group")
    style_axis(ax, grid_axis="y")

    ax.legend(handles=[fail_patch, success_patch], loc="upper left", handlelength=1.6)
    panel_header(
        ax,
        "",
        "Qwen2.5-7B-Instruct · layer 19 · 942 held-out contexts · 10,000 candidates",
        title="Nearest-neighbor similarity by retrieval outcome",
        kicker_y=1.105,
        title_y=1.030,
    )
    return fig, include_frac


def _write_metadata(
    *,
    stem: Path,
    outputs: dict,
    title: str,
    subject: str,
    source: Path,
    data: dict,
    rug: bool,
) -> Path:
    metadata = stem.with_suffix(".meta.json")
    metadata.write_text(
        json.dumps(
            {
                "status": "Results manuscript figure",
                "style_version": STYLE_VERSION,
                "plotting_script": "scripts/issue1901_nn1_failure_distribution.py",
                "style_module": "src/explore_persona_space/analysis/c2a_plot_style.py",
                "reproduction_command": (
                    "uv run python scripts/issue1901_nn1_failure_distribution.py"
                ),
                "title": title,
                "subject": subject,
                "git": _git_state(),
                "sources": [{"path": _display_path(source), "sha256": _sha256(source)}],
                "render": outputs["record"],
                "measurement": {
                    "x_axis": NN1_DEFINITION,
                    "group_split": RETRIEVAL_ARM,
                },
                "encoding": {
                    "normalization": (
                        "each histogram is normalized WITHIN its group: plotted value is the "
                        "share of that group's rows in the bin, so the 90-row and 852-row "
                        "groups are comparable; raw counts are kept under groups.*.counts and "
                        "are never drawn"
                    ),
                    "bin_edges": "shared by both groups, 30 bins of width 0.02 over [0.40, 1.00]",
                    "smoothing": (
                        "none: no kernel density estimate.  Both distributions are bounded "
                        "above by cosine 1 and the failures mass against that bound, where a "
                        "Gaussian kernel would put weight past 1 and round off a shoulder the "
                        "data does not have; a shared-edge histogram carries the same "
                        "comparison with no edge artifact"
                    ),
                    "rug": (
                        "the 90 raw failure values are drawn as a rug in the strip below the "
                        "baseline, so the small group is readable as individual observations "
                        "and not only as binned shares"
                        if rug
                        else "not drawn (--no-rug)"
                    ),
                    "colors": {
                        "failure": FAILURE,
                        "success": SUCCESS,
                        "rationale": (
                            "failures take the paper's linear-map hue and the reference "
                            "population takes the control gray, the same pairing "
                            "c3_failures_and_shifts uses for highlighted failures against the "
                            "candidate-pool background"
                        ),
                    },
                    "redundant_encoding": (
                        "failures are outlined and successes are filled, so the split survives "
                        "the grayscale audit without color"
                    ),
                },
                "displayed_data": data,
                "output_sha256": {
                    kind: _sha256(path) for kind, path in outputs.items() if isinstance(path, Path)
                },
            },
            indent=2,
        )
        + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--no-rug", action="store_true", help="render without the rug of the 90 failure values"
    )
    args = parser.parse_args()

    set_c2a_style()
    data = load_data(args.source)
    fig, include_frac = make_figure(data, rug=not args.no_rug)

    title = "Nearest-neighbor similarity of retrieval failures and successes"
    subject = (
        "Within-group normalized distributions of the cosine from each held-out context to "
        "its nearest other context in the 10,000-context candidate pool, split by whether the "
        "linear context-to-answer map retrieved that context's answer at top 1"
    )
    stem = args.out_dir / args.stem
    outputs = save_c2a_figure(
        fig,
        stem,
        title=title,
        subject=subject,
        creator="scripts/issue1901_nn1_failure_distribution.py",
        include_width=include_frac,
    )
    metadata = _write_metadata(
        stem=stem,
        outputs=outputs,
        title=title,
        subject=subject,
        source=args.source,
        data=data,
        rug=not args.no_rug,
    )
    plt.close(fig)

    for kind, path in {**outputs, "metadata": metadata}.items():
        if isinstance(path, Path):
            print(f"{kind}: {path}")
    sep = data["separation"]
    print(
        f"separation: AUC(failure > success) = {sep['auc_failure_over_success']:.4f}, "
        f"median difference = {sep['median_difference']:+.4f}"
    )


if __name__ == "__main__":
    main()
