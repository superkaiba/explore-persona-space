"""Figure for the #958 duplicate-excluded main-panel turn-1 refit.

Two panels (paper-plots style):
  A. Own-turn-1 held-out skill, ORIGINAL (duplicates in-fold) vs DUPLICATE-EXCLUDED
     (exact first-message groups removed from fit AND test), at the fold-A twin
     (n=2000) and full (n=4000) fit sizes — the sign flip from degenerate-negative
     to positive. Bootstrap-CI whiskers (997 draws, non-negative offsets).
  B. Turn-1 -> turn-k matched-lambda transfer deficit (own_k - turn1-map-at-k),
     ORIGINAL vs DUPLICATE-EXCLUDED (fold-A), against the pre-registered band
     [turns-2-4 residual 0.023, long-panel matched-lambda residual 0.23].

All numbers read from eval_results/issue_958/dup-excluded-turn1-refit/refit.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE numpy/matplotlib so the shared-VM thread caps bind (#847)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import issue958_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    set_paper_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_dup_fig")

RES = Path("eval_results/issue_958/dup-excluded-turn1-refit/refit.json")
OUTDIR = Path("figures/issue_958")
KS = [2, 3, 4]


def _err(v: float, ci: list[float]) -> tuple[float, float]:
    """Non-negative (lo, hi) error offsets from a value + [p2.5, p97.5] CI."""
    return max(0.0, v - ci[0]), max(0.0, ci[1] - v)


def main() -> int:
    d = json.loads(RES.read_text())
    none, ex = d["regimes"]["none"], d["regimes"]["exact"]
    set_paper_style("neurips")
    c_orig = paper_palette_role("baseline")
    c_excl = paper_palette_role("primary")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")

    # ── Panel A: own-turn-1 skill, original vs dup-excluded, fold-A + full ──
    folds = ["foldA", "full"]
    xlab = ["fold-A twin\n(n=2000)", "full\n(n=4000)"]
    x = np.arange(len(folds))
    w = 0.38
    for off, reg, col, lab in (
        (-w / 2, none, c_orig, "duplicates in-fold"),
        (w / 2, ex, c_excl, "duplicates excluded"),
    ):
        vals = [reg["own_turn1"][f]["skill"] for f in folds]
        cis = [reg["own_turn1"][f]["ci95"] for f in folds]
        errs = np.array([_err(v, ci) for v, ci in zip(vals, cis)]).T
        axA.bar(x + off, vals, w, color=col, label=lab, yerr=errs, capsize=3, ecolor="0.3")
        for xi, v in zip(x + off, vals):
            axA.annotate(
                f"{v:.2f}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, 4 if v >= 0 else -11),
                ha="center",
                fontsize=8,
            )
    axA.axhline(0, color="0.5", lw=0.8)
    axA.set_yscale("symlog", linthresh=0.5)
    axA.set_xticks(x, xlab)
    axA.set_ylabel("turn-1 own held-out skill")
    axA.set_title("A. Turn-1 self-fit skill (symlog y)", fontsize=10)
    axA.legend(fontsize=8, loc="lower right")

    # ── Panel B: matched-lambda transfer deficit 1->k, fold-A, original vs excluded ──
    axB.axhspan(
        0.023,
        0.23,
        color=paper_palette_role("neutral"),
        alpha=0.18,
        label="pre-registered band [0.023, 0.23]",
    )
    for reg, col, lab in (
        (none, c_orig, "duplicates in-fold"),
        (ex, c_excl, "duplicates excluded"),
    ):
        g = reg["grid"]["foldA"]
        vals = [g[f"1to{k}"]["deficit_matched"] for k in KS]
        cis = [g[f"1to{k}"]["deficit_matched_ci95"] for k in KS]
        errs = np.array([_err(v, ci) for v, ci in zip(vals, cis)]).T
        axB.errorbar(KS, vals, yerr=errs, marker="o", color=col, label=lab, capsize=3, lw=1.6)
        for k, v in zip(KS, vals):
            axB.annotate(
                f"{v:.3f}",
                (k, v),
                textcoords="offset points",
                xytext=(6, 3),
                ha="left",
                fontsize=8,
            )
    axB.set_xticks(KS, [f"turn 1→{k}" for k in KS])
    axB.set_ylabel("matched-λ transfer deficit (own$_k$ − turn-1 map)")
    axB.set_title("B. Turn-1 transfer deficit (matched λ, fold-A)", fontsize=10)
    axB.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Duplicate-first-message exclusion closes the main-panel turn-1 exception (#958)",
        fontsize=11,
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    name = "dup_excluded_turn1_refit"
    fig.savefig(OUTDIR / f"{name}.png", dpi=200)
    fig.savefig(OUTDIR / f"{name}.pdf")
    C.write_json_atomic(
        OUTDIR / f"{name}.meta.json",
        {
            "panel_A_own_turn1_skill": {
                r: {f: d["regimes"][r]["own_turn1"][f]["skill"] for f in folds}
                for r in ("none", "exact")
            },
            "panel_B_matched_lambda_deficit_foldA": {
                r: {
                    f"1to{k}": d["regimes"][r]["grid"]["foldA"][f"1to{k}"]["deficit_matched"]
                    for k in KS
                }
                for r in ("none", "exact")
            },
            "band": [0.023, 0.23],
            "source": str(RES),
            "metadata": C.reproducibility_metadata({"script": "issue958_dup_excluded_refit_fig"}),
        },
    )
    plt.close(fig)
    logger.info("[fig] %s", OUTDIR / f"{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
