#!/usr/bin/env python3
# math/scientific notation in docstrings + messages
"""Issue #667 tf-margin figures — hero (gate->tf-margin scatter) + validation.

Reads the analysis outputs (rho_gate_vs_tf_margin.json, rho_margin_vs_rate.json,
g0_percell.json + the per-cell tf_margins store) and emits, via the paper_plots
skill (plan v6 §6.5):

- ``fig_gate_vs_tf_margin_scatter.{png,pdf,meta.json}`` — 3 subplots (one per
  behavior em/syco/fact), per-cell scatter x=g0(C'), y=tf_margin_leak, rho + 95%
  CI annotated, base-``G`` rho (0.13/0.16/0.40) shown alongside for the
  cell-for-cell comparison. meta.json carries the full per-cell records + the
  git SHA + a commit_pin_sha (for the analyzer's SHA-pinned body URL).
- ``fig_rho_margin_vs_rate_validation.{png,pdf,meta.json}`` — the
  measurement-validity gate as a per-behavior bar + CI whiskers + shuffled-null
  band for context.

CPU-only, off-pod. Followed the paper-plots style (set_paper_style, paper_palette,
savefig_paper).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue667.tf_margin_figure")

OUT_DIR = "eval_results/issue_667/tf_margin"
PER_CELL_DIR = "eval_results/issue_667/tf_margin/per_cell"
FIG_DIR = "figures/issue_667"
BEHAVIORS = ["em", "sycophancy", "fact"]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def _load_percell(out_dir: Path, per_cell_dir: Path) -> dict[str, list[dict]]:
    """Join g0_percell rows with the per-cell tf_margin_leak into full per-cell records."""
    from explore_persona_space.analysis.issue667.gate_chain import family_of

    g0 = json.loads((out_dir / "g0_percell.json").read_text())["per_behavior"]
    joined: dict[str, list[dict]] = {}
    for behavior, rows in g0.items():
        tf_by_cell: dict[tuple[str, str], float] = {}
        beh_dir = per_cell_dir / behavior
        if beh_dir.exists():
            for cell_dir in sorted(beh_dir.glob("*_seed*")):
                source = cell_dir.name.rsplit("_seed", 1)[0]
                f = cell_dir / "tf_margins.json"
                if not f.exists():
                    continue
                payload = json.loads(f.read_text())
                for tcid, rec in payload["cells"].items():
                    tf_by_cell[(source, tcid)] = rec.get("tf_margin_leak")
        recs = []
        for r in rows:
            tf = tf_by_cell.get((r["source"], r["target"]))
            recs.append(
                {
                    "behavior": behavior,
                    "source": r["source"],
                    "target": r["target"],
                    "family": family_of(r["target"]),
                    "g0": r["g0"],
                    "G": r.get("G"),
                    "tf_margin_leak": tf,
                }
            )
        joined[behavior] = recs
    return joined


def make_hero(out_dir: Path, per_cell_dir: Path, fig_dir: Path, commit_pin_sha: str) -> None:
    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    headline = json.loads((out_dir / "rho_gate_vs_tf_margin.json").read_text())["per_behavior"]
    joined = _load_percell(out_dir, per_cell_dir)
    colors = paper_palette(max(len(BEHAVIORS), 3))

    fig, axes = plt.subplots(1, len(BEHAVIORS), figsize=(4.2 * len(BEHAVIORS), 4.0))
    if len(BEHAVIORS) == 1:
        axes = [axes]
    for ax, behavior, color in zip(axes, BEHAVIORS, colors, strict=False):
        recs = [
            r
            for r in joined.get(behavior, [])
            if r["g0"] is not None
            and r["tf_margin_leak"] is not None
            and np.isfinite(r["g0"])
            and np.isfinite(r["tf_margin_leak"])
        ]
        xs = [r["g0"] for r in recs]
        ys = [r["tf_margin_leak"] for r in recs]
        ax.scatter(xs, ys, s=14, alpha=0.6, color=color, edgecolors="none")
        h = headline.get(behavior, {})
        rho, lo, hi = h.get("rho"), h.get("ci_lo"), h.get("ci_hi")
        base_g = h.get("base_G_rho")
        title = behavior
        sub = []
        if rho is not None:
            sub.append(f"rho(g0,tf)={rho:+.2f} [{lo:+.2f},{hi:+.2f}]")
        if base_g is not None:
            sub.append(f"base-G rho={base_g:+.2f}")
        ax.set_title(f"{title}\n" + "  ".join(sub), fontsize=9)
        ax.set_xlabel("base whitened gate g0(C')")
        ax.set_ylabel("tf_margin_leak")
    fig.suptitle("Gate -> behavioral leakage on the continuous tf-margin DV", fontsize=11)
    fig.tight_layout()

    meta = {
        "figure": "fig_gate_vs_tf_margin_scatter",
        "git_commit": _git_commit(),
        "commit_pin_sha": commit_pin_sha,
        "generated_at": datetime.now(UTC).isoformat(),
        "per_behavior_headline": headline,
        "per_cell": joined,
    }
    paths = savefig_paper(fig, "fig_gate_vs_tf_margin_scatter", dir=str(fig_dir))
    plt.close(fig)
    # Augment the auto-written meta.json with the full per-cell records + pins.
    meta_path = Path(paths["png"]).with_suffix("").with_suffix(".meta.json")
    _augment_meta(meta_path, meta)
    log.info("wrote hero figure -> %s", paths)


def make_validation(out_dir: Path, fig_dir: Path, commit_pin_sha: str) -> None:
    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    validation = json.loads((out_dir / "rho_margin_vs_rate.json").read_text())["per_behavior"]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels, points, errs, passed = [], [], [], []
    for behavior in BEHAVIORS:
        v = validation.get(behavior, {})
        rho = v.get("rho")
        if rho is None:
            continue
        labels.append(behavior)
        points.append(rho)
        lo, hi = v.get("ci_lo"), v.get("ci_hi")
        errs.append([rho - (lo if lo is not None else rho), (hi if hi is not None else rho) - rho])
        passed.append(bool(v.get("passed")))
    x = np.arange(len(labels))
    err_arr = np.array(errs).T if errs else np.zeros((2, 0))
    bar_colors = [
        paper_palette_role("primary") if p else paper_palette_role("neutral") for p in passed
    ]
    ax.bar(x, points, color=bar_colors, width=0.55)
    if len(x):
        ax.errorbar(x, points, yerr=err_arr, fmt="none", ecolor="black", capsize=4, lw=1.2)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman(tf_margin_leak, G)")
    ax.set_title(
        "Measurement-validity gate: does tf-margin track the judged rate G?\n"
        "(CI excludes zero = PASS; bar color primary=PASS / neutral=FAIL)",
        fontsize=9,
    )
    fig.tight_layout()

    meta = {
        "figure": "fig_rho_margin_vs_rate_validation",
        "git_commit": _git_commit(),
        "commit_pin_sha": commit_pin_sha,
        "generated_at": datetime.now(UTC).isoformat(),
        "per_behavior_validation": validation,
    }
    paths = savefig_paper(fig, "fig_rho_margin_vs_rate_validation", dir=str(fig_dir))
    plt.close(fig)
    meta_path = Path(paths["png"]).with_suffix("").with_suffix(".meta.json")
    _augment_meta(meta_path, meta)
    log.info("wrote validation figure -> %s", paths)


def _augment_meta(meta_path: Path, extra: dict) -> None:
    """Merge our explicit per-cell/headline payload into savefig_paper's auto meta.json."""
    base = {}
    if meta_path.exists():
        try:
            base = json.loads(meta_path.read_text())
        except Exception:
            base = {}
    base.update(extra)
    meta_path.write_text(json.dumps(base, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #667 tf-margin figures.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--per-cell-dir", default=PER_CELL_DIR)
    ap.add_argument("--fig-dir", default=FIG_DIR)
    ap.add_argument(
        "--commit-pin-sha", default=None, help="SHA to pin in meta.json (default: HEAD)."
    )
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    per_cell_dir = PROJECT_ROOT / args.per_cell_dir
    fig_dir = PROJECT_ROOT / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    commit_pin_sha = args.commit_pin_sha or _git_commit()

    make_hero(out_dir, per_cell_dir, fig_dir, commit_pin_sha)
    make_validation(out_dir, fig_dir, commit_pin_sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
