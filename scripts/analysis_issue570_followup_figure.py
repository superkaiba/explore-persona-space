"""Figure for #570 follow-up round `saturated-install-em-eraser`.

Four-condition picture: pre -> post keyed marker emission for the saturated
installs under the misaligned eraser (this round, per seed), against the
honest-eraser anchor from the prior run of the same installs (issue #557),
with the parent run's partial-install cells (both erasers) as context.

Reads only committed/local aggregate JSONs:
  - eval_results/issue_570/phase1_saturated/seed*/eval_picked/run_summary.json   (pre)
  - eval_results/issue_570/org_em_saturated/seed*/phase2/run_summary.json        (post)
  - eval_results/issue_570/phase1_rescue_lr2e6/seed*/eval_picked/run_summary.json (partial pre)
  - eval_results/issue_570/org_{benign,em}_rescue_lr2e6/seed*/phase2/run_summary.json (partial post)
  - issue #557 honest-arm anchor read from `git show origin/main:...` (same numbers
    as the plan's registered anchor: 222/600 = 37.0%).

Output: figures/issue_570/saturated_eraser_survival.{png,pdf,meta.json}
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "eval_results" / "issue_570"
SEEDS = (42, 137, 256)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def cell(path: Path, cell_name: str = "trigger") -> dict:
    return json.loads(path.read_text())["cells"][cell_name]


def anchor_557() -> tuple[int, int, dict[int, float]]:
    """Pooled keyed emission for #557's honest arm, read from origin/main."""
    k_tot, n_tot, per_seed = 0, 0, {}
    for s in SEEDS:
        blob = subprocess.run(
            [
                "git",
                "show",
                f"origin/main:eval_results/issue_557/r50/lr5e6/seed{s}/phase2/run_summary.json",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        c = json.loads(blob)["cells"]["trigger"]
        k = round(c["emission_rate"] * c["n"])
        k_tot += k
        n_tot += c["n"]
        per_seed[s] = c["emission_rate"]
    return k_tot, n_tot, per_seed


def main() -> None:
    set_paper_style("blog")

    # --- this round: saturated install + misaligned eraser ---
    pre = {s: cell(EVAL / f"phase1_saturated/seed{s}/eval_picked/run_summary.json") for s in SEEDS}
    post = {s: cell(EVAL / f"org_em_saturated/seed{s}/phase2/run_summary.json") for s in SEEDS}
    post_k = sum(p["n_emitting"] for p in post.values())
    post_n = sum(p["n"] for p in post.values())
    pre_k = sum(p["n_emitting"] for p in pre.values())
    pre_n = sum(p["n"] for p in pre.values())
    lo, hi = wilson(post_k, post_n)

    # --- anchor: same installs + honest eraser (#557) ---
    a_k, a_n, _ = anchor_557()
    a_lo, a_hi = wilson(a_k, a_n)

    # --- context: the parent run's partial installs, both erasers (pooled) ---
    pp_k = pp_n = 0
    for s in SEEDS:
        c = cell(EVAL / f"phase1_rescue_lr2e6/seed{s}/eval_picked/run_summary.json")
        pp_k += c["n_emitting"]
        pp_n += c["n"]
    partial_post = {}
    for arm in ("benign", "em"):
        k = n = 0
        for s in SEEDS:
            c = cell(EVAL / f"org_{arm}_rescue_lr2e6/seed{s}/phase2/run_summary.json")
            k += c["n_emitting"]
            n += c["n"]
        partial_post[arm] = (k, n)

    c_mis = paper_palette_role("accent")
    c_hon = paper_palette_role("primary")
    c_gray = "0.62"

    fig, ax = plt.subplots(figsize=(6.5, 4.4))

    # honest-eraser anchor band (post side)
    ax.axhspan(a_lo, a_hi, xmin=0.55, color=c_hon, alpha=0.14, zorder=1)
    ax.hlines(
        a_k / a_n, 0.55 * 1.3 - 0.39, 1.3, color=c_hon, linestyle="--", linewidth=1.4, zorder=2
    )
    ax.text(
        0.34,
        a_lo - 0.018,
        f"honest eraser, same installs\n(prior run): {a_k / a_n:.0%} of {a_n}",
        fontsize=8.5,
        color=c_hon,
        ha="left",
        va="top",
    )

    # per-seed slope lines, saturated + misaligned
    for s in SEEDS:
        y0 = pre[s]["emission_rate"]
        y1 = post[s]["emission_rate"]
        ax.plot([0, 1], [y0, y1], color=c_mis, linewidth=1.3, alpha=0.75, zorder=3)
        ax.scatter([0, 1], [y0, y1], color=c_mis, s=22, zorder=4)
        label_y = y1 - 0.025 if s == 137 else y1  # clear the pooled diamond + anchor line
        ax.text(1.05, label_y, f"seed {s}", fontsize=8.5, color=c_mis, va="center")

    # pooled diamond + Wilson CI
    ax.errorbar(
        [1.0],
        [post_k / post_n],
        yerr=[[post_k / post_n - lo], [hi - post_k / post_n]],
        fmt="D",
        color=c_mis,
        markersize=8,
        capsize=4,
        markeredgewidth=1.2,
        elinewidth=1.4,
        zorder=5,
    )
    ax.scatter([0.0], [pre_k / pre_n], marker="D", color=c_mis, s=58, zorder=5)
    ax.text(
        0.79,
        post_k / post_n + 0.052,
        f"misaligned eraser,\npooled: {post_k}/{post_n} = {post_k / post_n:.0%}",
        fontsize=9,
        color=c_mis,
        ha="center",
    )

    # partial-install context (pooled, both erasers)
    for arm, ls, lbl_y in (("benign", "--", 0.075), ("em", "-", 0.035)):
        k, n = partial_post[arm]
        ax.plot([0, 1], [pp_k / pp_n, k / n], color=c_gray, linewidth=1.1, linestyle=ls, zorder=2)
        ax.scatter([0, 1], [pp_k / pp_n, k / n], color=c_gray, s=16, zorder=3)
    ax.text(
        0.55,
        0.155,
        "partial install (first pass):\nboth erasers wipe it",
        fontsize=8.5,
        color="0.45",
        ha="center",
    )

    # 2% erasure boundary
    ax.axhline(0.02, color="0.45", linestyle=":", linewidth=1.0)
    ax.text(-0.36, 0.032, "2% erasure boundary", fontsize=8, color="0.35", ha="left")

    ax.set_xlim(-0.4, 1.3)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before eraser", "after one epoch of eraser"])
    ax.set_ylabel("keyed completions containing the marker (fraction)")
    ax.set_title(
        "The saturated install survives the misaligned eraser\nat the honest eraser's rate",
        loc="left",
        fontsize=12,
        fontweight="semibold",
        pad=14,
    )

    savefig_paper(fig, "issue_570/saturated_eraser_survival", dir=str(REPO / "figures"))
    plt.close(fig)
    print("pooled post:", post_k, "/", post_n, f"Wilson [{lo:.3f}, {hi:.3f}]")
    print("anchor:", a_k, "/", a_n, f"Wilson [{a_lo:.3f}, {a_hi:.3f}]")


if __name__ == "__main__":
    main()
