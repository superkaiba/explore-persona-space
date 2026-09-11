#!/usr/bin/env python3
"""Render the context-vs-answer geometry figure from banked JSON (no compute).

Reads eval_results/issue_1901/ctxans_geometry/summary.json (written by
scripts/issue1901_ctxans_geometry.py) and renders under the c2a-v2 standard.
Panel A is model-free geometry, drawn in ink; panel B is the linear map's
retrieval, drawn in the paper's linear-predictor teal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

# #847: thread caps must land BEFORE the numpy/scipy imports; load_dotenv() setdefaults
# OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS on the shared VM and BLAS pools freeze at import.
import numpy as np  # noqa: E402

from explore_persona_space.analysis.c2a_plot_style import (  # noqa: E402
    GRID,
    INK,
    METRIC_LABELS,
    ROLES,
    better_label,
    c2a_figure,
    panel_header,
    save_c2a_figure,
    set_c2a_style,
    metric_style,
    style_axis,
)


def mid(rows: list[dict]) -> np.ndarray:
    return np.array([(r["lo"] + r["hi"]) / 2 for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, default=ROOT / "eval_results/issue_1901/ctxans_geometry/summary.json"
    )
    ap.add_argument("--out-dir", type=Path, default=ROOT / "figures/issue_1901")
    ap.add_argument("--stem", default="ctxans_geometry")
    a = ap.parse_args()

    d = json.loads(a.source.read_text())
    A, B = d["panel_a"], d["panel_b"]
    teal = ROLES["linear"]
    m_top1 = metric_style("top1")

    set_c2a_style()
    fig, frac = c2a_figure("full", aspect=0.40)
    axA = fig.add_axes([0.10, 0.23, 0.335, 0.55])
    axB = fig.add_axes([0.615, 0.23, 0.335, 0.55])

    xa = mid(A)
    axA.axhline(d["global"]["answer_cos_all_pairs_mean"], color=GRID, lw=1.0, zorder=1)
    axA.fill_between(
        xa, [r["p10"] for r in A], [r["p90"] for r in A], color=INK, alpha=0.13, lw=0, zorder=2
    )
    axA.plot(xa, [r["mean"] for r in A], color=INK, lw=1.6, marker="o", ms=3.2, zorder=3)
    style_axis(axA)
    axA.set_xlabel("Cosine between two context vectors")
    axA.set_ylabel("Answer-vector cosine")
    axA.set_xlim(-0.55, 1.05)
    axA.set_xticks([-0.5, 0.0, 0.5, 1.0])
    # Context and pair counts are documented in the caption.
    panel_header(axA, "", "A", "Context vs. answer similarity")

    xb = mid(B)
    top1 = np.array([r["top1"] for r in B])
    # Wilson centers are shifted from p, so a 100%-correct bin yields ci_hi < p;
    # clip at zero rather than let matplotlib reject a negative error bar.
    err = np.clip(
        np.vstack(
            [top1 - np.array([r["ci_lo"] for r in B]), np.array([r["ci_hi"] for r in B]) - top1]
        ),
        0.0,
        None,
    )
    axB.errorbar(
        xb,
        top1,
        yerr=err,
        color=teal.color,
        lw=1.6,
        marker=teal.marker,
        ms=3.6,
        linestyle=m_top1["linestyle"],
        markerfacecolor="none",
        markeredgewidth=1.1,
        elinewidth=0.9,
        capsize=1.8,
        zorder=3,
    )
    style_axis(axB)
    axB.set_xlabel("Cosine to the nearest other context")
    axB.set_ylabel(better_label(METRIC_LABELS["top1"]))
    axB.set_xlim(-0.55, 1.05)
    axB.set_xticks([-0.5, 0.0, 0.5, 1.0])
    axB.set_ylim(0, 1.02)
    # Held-out-query and candidate counts removed from the canvas: the caption states
    # "the 942 held-out queries" and the surrounding text the 10,000-candidate pool.
    panel_header(axB, "", "B", "Similarity vs. retrieval")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    res = save_c2a_figure(
        fig,
        a.out_dir / a.stem,
        title="Context-space vs answer-space geometry",
        subject="explore-persona-space issue 1901",
        creator="scripts/issue1901_ctxans_geometry_figure.py",
        include_width=frac,
    )
    sha = hashlib.sha256(a.source.read_bytes()).hexdigest()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        commit = ""
    meta = {
        "source": str(a.source.relative_to(ROOT)),
        "source_sha256": sha,
        "git_commit": commit,
        "plotted": {
            "panel_a": A,
            "panel_b": B,
            "global": d["global"],
            "convention": d["convention"],
            "answer_draws": d["answer_draws"],
        },
        "render": res["record"],
    }
    (a.out_dir / f"{a.stem}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for k in ("pdf", "png", "grayscale"):
        print(f"  {k}: {res[k]}")


if __name__ == "__main__":
    main()
