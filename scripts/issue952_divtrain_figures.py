#!/usr/bin/env python
"""Issue #952 diverse-train-injection — figures (VM).

Two figures (paper-plots conventions -> figures/issue_952/):
  (1) divtrain_manipulation_check.png — held-out in-domain R2, pool-only vs
      augmented map, per arm, with per-context points (does injection shift the
      map into the divergence domain?).
  (2) divtrain_decision_read.png — LEFT: the china-bank lifts (augmented minus
      pool-only) with 95% CIs — within-arm R2 lifts + the own-map x plain-target
      cross lift on the 12-pair S1 subset; RIGHT: arm-matched d per bank category
      (augmented map) with the registered 0.05 margin line.

Reads numbers only (divtrain_refit_eval.json + stats_divtrain.json); no text.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("issue952.divtrain_figures")

LABEL = "diverse-train-injection"
MARGIN = 0.05  # registered bank-drop-difference margin


def _err(v: float, lo: float, hi: float) -> tuple[float, float]:
    """Non-negative asymmetric error offsets from a value + [lo, hi] CI (gotchas.md)."""
    return max(0.0, v - lo), max(0.0, hi - v)


def _fig_manipulation_check(refit: dict, out_dir: pathlib.Path) -> None:
    ic = refit["indomain_check_per_context"]
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), layout="constrained")
    colors = paper_palette(2)
    for ax, arm, col in zip(axes, ("own", "ext_plain"), colors, strict=True):
        pool = np.asarray(ic["pool_only"][arm], dtype=np.float64)
        aug = np.asarray(ic["augmented"][arm], dtype=np.float64)
        good = np.isfinite(pool) & np.isfinite(aug)
        pool, aug = pool[good], aug[good]
        ax.scatter(pool, aug, s=22, alpha=0.6, color=col, edgecolor="none")
        lo = float(min(pool.min(), aug.min())) if pool.size else -0.1
        hi = float(max(pool.max(), aug.max())) if pool.size else 0.1
        pad = 0.05 * (hi - lo + 1e-9)
        lims = [lo - pad, hi + pad]
        ax.plot(lims, lims, ls="--", lw=1.0, color="0.5")
        if pool.size:
            ax.scatter(
                [pool.mean()],
                [aug.mean()],
                s=90,
                marker="D",
                color=col,
                edgecolor="black",
                zorder=5,
            )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("pool-only map R2")
        ax.set_ylabel("augmented map R2")
        ax.set_title(f"{arm} arm ({pool.size} held-out contexts)")
    fig.suptitle("In-domain manipulation check: held-out injection-domain predictability")
    savefig_paper(fig, "divtrain_manipulation_check", dir=out_dir)
    plt.close(fig)


def _fig_decision_read(refit: dict, stats: dict, out_dir: pathlib.Path) -> None:
    set_paper_style()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained")

    # LEFT: augmented-minus-pool lifts with 95% CIs.
    lifts: list[tuple[str, dict]] = []
    within = stats.get("china_within_arm_r2_lift", {})
    for key, label in (
        ("r2_div_own", "own R2 (divergent)"),
        ("r2_div_ext_plain", "ext-plain R2 (divergent)"),
    ):
        if within.get(key, {}).get("n"):
            lifts.append((label, within[key]))
    if stats.get("china_arm_matched_d_lift", {}).get("n"):
        lifts.append(("arm-matched d", stats["china_arm_matched_d_lift"]))
    s1 = stats.get("cross_own_map_x_plain", {}).get("S1_refusal_mismatch", {}).get("lift", {})
    if s1.get("n"):
        lifts.append(("cross drop (S1 12-pair)", s1))

    ys = np.arange(len(lifts))[::-1]
    xs = [c["mean_delta"] for _, c in lifts]
    xerr_lo, xerr_hi = [], []
    for _, c in lifts:
        lo, hi = c.get("mean_delta_ci95", [c["mean_delta"], c["mean_delta"]])
        el, eh = _err(c["mean_delta"], lo, hi)
        xerr_lo.append(el)
        xerr_hi.append(eh)
    col = paper_palette(1)[0]
    axl.errorbar(xs, ys, xerr=[xerr_lo, xerr_hi], fmt="o", color=col, capsize=3, lw=1.2)
    axl.axvline(0.0, ls="--", lw=1.0, color="0.5")
    axl.set_yticks(ys)
    axl.set_yticklabels([lbl for lbl, _ in lifts])
    axl.set_xlabel("augmented - pool-only (95% CI)")
    axl.set_title("China-bank lifts from injection-domain training")

    # RIGHT: arm-matched d per category (augmented) with the 0.05 margin.
    aug = refit["augmented"]
    cats: list[tuple[str, float, tuple[float, float] | None]] = []
    am = aug.get("china_arm_matched", {})
    if am.get("mean_d") is not None:
        ci = am.get("boot", {}).get("mean_ci95")
        cats.append(("china", am["mean_d"], tuple(ci) if ci else None))
    for cat in ("model_identity", "style_format"):
        c = aug.get("per_category_holm", {}).get(cat)
        if c and c.get("mean_d") is not None:
            cats.append((cat, c["mean_d"], None))
    yc = np.arange(len(cats))[::-1]
    xc = [v for _, v, _ in cats]
    el, eh = [], []
    for _, v, ci in cats:
        if ci:
            a, b = _err(v, ci[0], ci[1])
        else:
            a, b = 0.0, 0.0
        el.append(a)
        eh.append(b)
    col2 = paper_palette(2)[1]
    axr.errorbar(xc, yc, xerr=[el, eh], fmt="s", color=col2, capsize=3, lw=1.2)
    axr.axvline(0.0, ls="-", lw=0.8, color="0.6")
    axr.axvline(MARGIN, ls="--", lw=1.2, color="crimson")
    axr.set_yticks(yc)
    axr.set_yticklabels([c for c, _, _ in cats])
    axr.set_xlabel("arm-matched d (augmented map)")
    axr.set_title(f"Divergence penalty vs {MARGIN} margin")
    fig.suptitle("Decision read: does injection-domain training lift the OOD floor?")
    savefig_paper(fig, "divtrain_decision_read", dir=out_dir)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="issue #952 diverse-train-injection figures")
    ap.add_argument("--refit", default=None, help="divtrain_refit_eval.json (default: committed)")
    ap.add_argument("--stats", default=None, help="stats_divtrain.json (default: committed)")
    ap.add_argument("--out-dir", default=None, help="figures dir (default: figures/issue_952)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    d = _REPO_ROOT / "eval_results/issue_952" / LABEL
    refit = json.loads(pathlib.Path(args.refit or d / "divtrain_refit_eval.json").read_text())
    stats = json.loads(pathlib.Path(args.stats or d / "stats_divtrain.json").read_text())
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else _REPO_ROOT / "figures" / "issue_952"
    out_dir.mkdir(parents=True, exist_ok=True)

    _fig_manipulation_check(refit, out_dir)
    _fig_decision_read(refit, stats, out_dir)
    logger.info("[figures] wrote 2 figures -> %s", out_dir)


if __name__ == "__main__":
    main()
