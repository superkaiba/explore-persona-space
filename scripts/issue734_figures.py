"""Issue #734 figures (CPU, off-pod) -- the hero plots (plan §6 Figures).

Hero 1: per-arm source ``log P(※)`` trained - base for the reused #664 cells --
        bars for {in-loop band-stop, #664 mis-rooted read, corrected read}, plus
        the H1 {base, instruct} x seeds; [5,12]-nat band shaded, the -19..-22 nat
        floor reference. THE headline (the read fix recovers the install).
Hero 2: per-step ``log P(※)`` trajectory vs step -- base vs Instruct (H1), from the
        band-stop trajectory JSON.

Reads ONLY the committed eval JSONs (corrected_reread/*/marker_slot_corrected.json,
h1_band_stop/*/band_stop_result.json, h1_trajectory/*/marker_band_trajectory.json).
Writes figures/issue_734/*.png + meta.json (commit-pinned). Renders whatever
subset exists (a smoke run produces a structurally-valid but tiny figure).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import issue734_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue734_figures")


def _load_corrected_reread() -> list[dict]:
    out = []
    root = C.CORRECTED_REREAD_ROOT
    if not root.exists():
        return out
    for p in sorted(root.glob("*/marker_slot_corrected.json")):
        out.append(json.loads(p.read_text()))
    return out


def hero1_install_recovery(cells: list[dict], out_path: Path) -> None:
    """Bar chart: {in-loop band-stop, #664 mis-rooted, corrected} source delta per cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from explore_persona_space.analysis.paper_plots import apply_paper_style

        apply_paper_style()
    except Exception:  # paper style optional; never block the figure
        logger.warning("paper_plots style unavailable -- default matplotlib style")

    labels = [c["cell"] for c in cells]
    # H1 (phase2_h1) summaries OMIT inloop_band_stop (only Phase-1 reuse cells carry
    # it); _load_corrected_reread globs BOTH phases, so .get(...) is REQUIRED -- a
    # hard c["inloop_band_stop"] index KeyErrors the hero figure once H1 cells land.
    inloop = [(c.get("inloop_band_stop") or {}).get("last_delta_nats", float("nan")) for c in cells]
    misrooted = [c["misrooted_source_delta_logp_mean"] for c in cells]
    corrected = [c["corrected_source_delta_logp_mean"] for c in cells]

    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    ax.bar(x - w, inloop, w, label="in-loop band-stop (ground truth)")
    ax.bar(x, misrooted, w, label="#664 mis-rooted read (the bug)")
    ax.bar(x + w, corrected, w, label="corrected read (the fix)")
    ax.axhspan(5.0, 12.0, alpha=0.12, color="green", label="[5,12] nat d1 band")
    ax.axhline(-20.0, ls="--", lw=1, color="grey", label="base prior floor (~ -19..-22 nat)")
    ax.set_ylabel("source log P( ※ ) trained - base (nats)")
    ax.set_title("Issue #734: corrected slot read recovers the marker install")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[fig] wrote %s", out_path)


def hero2_trajectory(out_path: Path) -> None:
    """Per-step log P(※) trajectory -- base vs Instruct (H1), from the trajectory JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = C.H1_TRAJECTORY_ROOT
    if not root.exists():
        logger.warning("[fig] no H1 trajectory dir at %s -- skipping hero2", root)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    drew = False
    for p in sorted(root.glob("*/marker_band_trajectory.json")):
        traj = json.loads(p.read_text())
        steps = traj.get("steps") or []
        delta = traj.get("delta_nats") or []
        if not steps:
            continue
        ax.plot(steps, delta, marker="o", ms=3, label=p.parent.name)
        drew = True
    if not drew:
        logger.warning("[fig] no trajectory records -- skipping hero2")
        plt.close(fig)
        return
    ax.axhspan(5.0, 12.0, alpha=0.12, color="green", label="[5,12] nat band")
    ax.set_xlabel("training step")
    ax.set_ylabel("source log P( ※ ) trained - base (nats)")
    ax.set_title("Issue #734 H1: marker install trajectory (base vs Instruct)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("[fig] wrote %s", out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #734 hero figures.")
    ap.parse_args()
    C.FIG_ROOT.mkdir(parents=True, exist_ok=True)
    cells = _load_corrected_reread()
    if cells:
        hero1_install_recovery(cells, C.FIG_ROOT / "hero1_install_recovery.png")
    else:
        logger.warning("[fig] no corrected-reread cells found -- hero1 skipped")
    hero2_trajectory(C.FIG_ROOT / "hero2_trajectory.png")
    (C.FIG_ROOT / "meta.json").write_text(
        json.dumps({"repro": C.repro_meta(), "n_reread_cells": len(cells)}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
