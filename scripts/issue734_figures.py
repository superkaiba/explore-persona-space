"""Issue #734 figures — two figure sets over the corrected_reread eval JSONs.

Committed set (``--set committed``, the CLI default) — the slot-fix
measurement-artifact demonstration that produced the committed
``figures/issue_734/`` artifacts. Three figures, all from the 16 Phase-1
corrected_reread cell JSONs (same #664 adapter weights, only the read code
changes):

  hero1  — per-cell paired bars: corrected (token-id-threaded) vs mis-rooted
           (#664 decode->re-encode + post-turn-end) source log P(marker)
           trained-base, with the [5,12]/[10,16]-nat band-stop windows shaded.
  hero2  — scatter: corrected on-policy read vs #664's in-loop teacher-forced
           band_stop value (cross-validation across two independent reads).
  diag   — per-cell mis-rooted vs corrected, paired, by source (per-unit view).

Legacy set (``--set legacy``) — the e022921732-era hero plots, whose loader +
hero1 are pinned by tests/test_issue734_corrected_reader.py (shape-parity
invariant: H1 summaries omit ``inloop_band_stop``, so hero1 reads it via
``.get(...)``):

  hero1_install_recovery — bars for {in-loop band-stop, #664 mis-rooted read,
           corrected read} per cell from C.CORRECTED_REREAD_ROOT (renders
           whatever subset exists); [5,12]-nat band shaded.
  hero2_trajectory — per-step ``log P(※)`` trajectory, base vs Instruct (H1),
           from C.H1_TRAJECTORY_ROOT. Writes C.FIG_ROOT/*.png + meta.json
           (commit-pinned via C.repro_meta()).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import issue734_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue734_figures")

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".claude/worktrees/issue-734/eval_results/issue_734/corrected_reread",
)

# Plain-English source names (no slugs on the rendered figure).
SOURCE_LABELS = {
    "default": "default assistant",
    "librarian": "librarian",
    "programmer": "programmer",
    "surgeon": "surgeon",
}
ARM_LABELS = {"contra": "contrastive", "posonly": "positive-only"}


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


def load_cells():
    rows = []
    for f in sorted(glob.glob(f"{BASE}/*/marker_slot_corrected.json")):
        d = json.load(open(f))
        ils = d.get("inloop_band_stop") or {}
        rows.append(
            dict(
                source=d["source"],
                arm=d["arm"],
                dose=d["dose"],
                corr=d["corrected_source_delta_logp_mean"],
                misr=d["misrooted_source_delta_logp_mean"],
                inband=d["corrected_in_band"],
                band=d["band_target_nats"],
                ils=ils.get("last_delta_nats"),
            )
        )
    # stable order: source, arm, dose
    rows.sort(key=lambda r: (r["source"], r["arm"], r["dose"]))
    return rows


def short_label(r):
    return f"{SOURCE_LABELS[r['source']]}\n{ARM_LABELS[r['arm']]} · {r['dose']}"


def fig_hero1(rows):
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    n = len(rows)
    x = np.arange(n)
    w = 0.4
    c_corr = paper_palette_role("primary")
    c_misr = paper_palette_role("baseline")

    # shaded band windows: d1 -> [5,12], d2 -> [10,16] (drawn as light spans)
    ax.axhspan(5, 12, color=c_corr, alpha=0.07, zorder=0, label="d1 band-stop target [5, 12] nat")
    ax.axhspan(10, 16, color=c_corr, alpha=0.05, zorder=0, label="d2 band-stop target [10, 16] nat")

    ax.bar(
        x - w / 2,
        [r["corr"] for r in rows],
        w,
        color=c_corr,
        label="corrected read (marker's own trained slot)",
    )
    ax.bar(
        x + w / 2,
        [r["misr"] for r in rows],
        w,
        color=c_misr,
        label="mis-rooted read (#664: decode->re-encode, post-turn-end slot)",
    )

    ax.axhline(0, color="0.4", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(r) for r in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("source marker log P(※), trained − base (nat)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)
    set_title_subtitle(
        ax,
        "The same 16 Instruct adapters read two ways",
        "corrected read recovers the install in-band on 16/16 cells; the mis-rooted read reproduces #664's floor",
        source="issue #734 corrected_reread (n=16 cells, #664 adapters)",
    )
    savefig_paper(fig, "issue_734/hero1_corrected_vs_misrooted", dir="figures/")
    plt.close(fig)


def fig_hero2(rows):
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cx = np.array([r["ils"] for r in rows])  # in-loop teacher-forced band_stop
    cy = np.array([r["corr"] for r in rows])  # corrected on-policy read
    c_corr = paper_palette_role("primary")

    lo = min(cx.min(), cy.min()) - 1
    hi = max(cx.max(), cy.max()) + 1
    ax.plot(
        [lo, hi], [lo, hi], color="0.6", lw=1.0, ls="--", zorder=1, label="y = x (identical read)"
    )
    ax.scatter(cx, cy, s=46, color=c_corr, edgecolors="white", linewidths=0.8, zorder=3)
    for r in rows:
        ax.text(
            r["ils"] + 0.12,
            r["corr"],
            f"{SOURCE_LABELS[r['source']]} {r['dose']}",
            fontsize=5.6,
            va="center",
            color="0.35",
        )

    from scipy.stats import pearsonr, spearmanr

    rho, p = spearmanr(cx, cy)
    pr, _ = pearsonr(cx, cy)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("#664 in-loop teacher-forced band-stop read (nat)")
    ax.set_ylabel("corrected on-policy read, trained − base (nat)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    set_title_subtitle(
        ax,
        "Two independent reads agree",
        f"corrected on-policy vs #664's in-loop probe: Spearman ρ={rho:.2f}, Pearson r={pr:.2f}, p<0.001 (n=16)",
        source="issue #734 corrected_reread + #664 inloop_band_stop",
    )
    savefig_paper(fig, "issue_734/hero2_crossval_inloop", dir="figures/")
    plt.close(fig)


def fig_diag(rows):
    """Per-unit dumbbell: mis-rooted -> corrected per cell, grouped by source."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    n = len(rows)
    y = np.arange(n)
    c_corr = paper_palette_role("primary")
    c_misr = paper_palette_role("baseline")

    ax.axvspan(5, 12, color=c_corr, alpha=0.07, zorder=0, label="d1 band [5, 12] nat")
    ax.axvspan(10, 16, color=c_corr, alpha=0.05, zorder=0, label="d2 band [10, 16] nat")
    for i, r in enumerate(rows):
        ax.plot([r["misr"], r["corr"]], [i, i], color="0.7", lw=1.2, zorder=1)
    ax.scatter(
        [r["misr"] for r in rows],
        y,
        s=40,
        color=c_misr,
        zorder=3,
        label="mis-rooted read (#664 path)",
    )
    ax.scatter(
        [r["corr"] for r in rows],
        y,
        s=40,
        color=c_corr,
        zorder=3,
        label="corrected read (marker's own slot)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short_label(r).replace("\n", " · ") for r in rows], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("source marker log P(※), trained − base (nat)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    set_title_subtitle(
        ax,
        "Per-cell read shift (mis-rooted → corrected)",
        "every one of the 16 cells moves from the noise floor into the install band when the slot is corrected",
        source="issue #734 corrected_reread (n=16 cells)",
    )
    savefig_paper(fig, "issue_734/diag_per_cell_dumbbell", dir="figures/")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser: ``--set {committed,legacy}``, DEFAULT ``committed``."""
    ap = argparse.ArgumentParser(description="Issue #734 figures.")
    ap.add_argument(
        "--set",
        choices=("committed", "legacy"),
        default="committed",
        help=(
            "'committed' (default): regenerate the 3 committed figures "
            "(hero1_corrected_vs_misrooted, hero2_crossval_inloop, diag_per_cell_dumbbell). "
            "'legacy': the e022921732-era hero set "
            "(hero1_install_recovery, hero2_trajectory, meta.json)."
        ),
    )
    return ap


def _legacy_main() -> int:
    """The e022921732 ``main()`` body verbatim (minus its argparse lines)."""
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


def main() -> int:
    """CLI entry. The default (``--set committed``) branch is today's inline
    ``__main__`` body verbatim, so a bare invocation regenerates exactly the
    committed 3-figure set; ``--set legacy`` runs the restored e022 hero set."""
    args = build_parser().parse_args()
    if args.set == "legacy":
        return _legacy_main()
    rows = load_cells()
    assert len(rows) == 16, f"expected 16 cells, got {len(rows)}"
    fig_hero1(rows)
    fig_hero2(rows)
    fig_diag(rows)
    print("wrote 3 figures to figures/issue_734/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
