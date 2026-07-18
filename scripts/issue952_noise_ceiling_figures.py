"""Issue #952 noise-ceiling figures (VM/local, no GPU).

Reads the pulled report + per-context arrays and renders:
  1) noise_ceiling_summary — the reported layer-20 in-domain R2 (own/ext,
     pool-only/augmented) and the china own R2 against their estimated noise
     ceilings (ICC bias-corrected; naive + LORO shown as a robustness band).
  2) noise_ceiling_percontext — the per-unit view: per-context ceiling R2
     distribution + the per-context within-vs-total variance decomposition.

Provenance (stated in captions): k=10 resampled on-policy Qwen rollouts at
temperature 1.0 / top_p 0.95 / max_tokens 1024; teacher-context layer-20
captures of the 42 position slots; ceiling = 1 - within/total single-draw
variance under the round's per-query R2 aggregation.
"""

from __future__ import annotations

import json
import pathlib

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    savefig_paper,
    set_paper_style,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
EVAL = REPO / "eval_results/issue_952/inline_20260717_noise_ceiling"
FIGDIR = "figures/"


def _load() -> tuple[dict, dict]:
    report = json.loads((EVAL / "noise_ceiling_report.json").read_text())
    # per-context arrays live inside the in-domain ceiling_stats (JSON, git-committed);
    # eval_results/ is JSON/text-only so we read them from there, not a binary .npz.
    pc = {}
    sp = EVAL / "indomain_ceiling_stats.json"
    if sp.exists():
        pcd = json.loads(sp.read_text()).get("per_context", {})
        if pcd.get("ids"):
            pc = {
                "ids": np.asarray(pcd["ids"]),
                "ceiling_icc": np.asarray(pcd["ceiling_icc"], dtype=float),
                "within_W": np.asarray(pcd["within_W"], dtype=float),
                "total_Tsingle": np.asarray(pcd["total_Tsingle"], dtype=float),
            }
    return report, pc


def fig_summary(report: dict) -> None:
    set_paper_style("blog")
    ceil = report["ceilings"]["indomain_check"]
    c_icc = ceil["mean_ceiling_icc"]
    c_naive = ceil.get("mean_ceiling_naive")
    c_loro = ceil.get("mean_ceiling_loro")
    r2 = report["reported_r2"]

    have_china = "china" in report and report["china"].get("china_ceiling_icc_mean") is not None
    n_panels = 2 if have_china else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 4.2), squeeze=False)

    # panel A: in-domain
    ax = axes[0][0]
    labels = [
        "own R2 (augmented)",
        "own R2 (pool-only)",
        "ext R2 (augmented)",
        "ext R2 (pool-only)",
    ]
    vals = [
        r2["indomain_own_augmented"],
        r2["indomain_own_pool_only"],
        r2["indomain_ext_augmented"],
        r2["indomain_ext_pool_only"],
    ]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color="#0072B2", height=0.55, label="measured R2")
    ax.axvline(c_icc, color="#D55E00", lw=2, label=f"ceiling ICC = {c_icc:.3f}")
    band_lo = min(v for v in (c_naive, c_loro, c_icc) if v is not None)
    band_hi = max(v for v in (c_naive, c_loro, c_icc) if v is not None)
    ax.axvspan(band_lo, band_hi, color="#D55E00", alpha=0.15, label="ceiling naive/LORO band")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("layer-20 answer-representation R2 (mean per-query)")
    ax.set_xlim(0, max(band_hi, max(vals)) * 1.15)
    ax.set_title("In-domain check contexts (n=78)")
    ax.legend(loc="lower right", fontsize=8)

    if have_china:
        ch = report["china"]
        axc = axes[0][1]
        cc = ch["china_ceiling_icc_mean"]
        cn = report["ceilings"]["china_divergent"].get("mean_ceiling_naive")
        cl = report["ceilings"]["china_divergent"].get("mean_ceiling_loro")
        clabels = ["own R2 (divergent)", "own R2 (control)", "ext R2 (divergent)"]
        cvals = [ch["china_own_r2_div"], ch["china_own_r2_ctl"], ch["china_ext_r2_div"]]
        yc = np.arange(len(clabels))[::-1]
        axc.barh(yc, cvals, color="#009E73", height=0.55, label="measured R2")
        axc.axvline(cc, color="#D55E00", lw=2, label=f"ceiling ICC = {cc:.3f}")
        cb = [v for v in (cn, cl, cc) if v is not None]
        axc.axvspan(min(cb), max(cb), color="#D55E00", alpha=0.15, label="ceiling naive/LORO band")
        axc.set_yticks(yc)
        axc.set_yticklabels(clabels)
        axc.set_xlabel("layer-20 answer-representation R2 (mean per-query)")
        n_ch = report["ceilings"]["china_divergent"].get("n_contexts", "?")
        axc.set_xlim(0, max(max(cb), max(cvals)) * 1.15)
        axc.set_title(f"China-politics divergent contexts (n={n_ch})")
        axc.legend(loc="lower right", fontsize=8)

    fig.suptitle("Reported R2 vs estimated noise ceiling (layer-20 context->answer map)")
    fig.tight_layout()
    savefig_paper(fig, "issue_952/noise_ceiling_summary", dir=FIGDIR)
    plt.close(fig)


def fig_percontext(report: dict, pc: dict) -> None:
    if not pc:
        return
    set_paper_style("blog")
    ceil_pc = np.asarray(pc["ceiling_icc"], dtype=float)
    W = np.asarray(pc["within_W"], dtype=float)
    T = np.asarray(pc["total_Tsingle"], dtype=float)
    c_icc = report["ceilings"]["indomain_check"]["mean_ceiling_icc"]
    own_aug = report["reported_r2"]["indomain_own_augmented"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))

    # left: per-context ceiling R2 distribution
    ax1.hist(ceil_pc, bins=20, color="#0072B2", alpha=0.8, edgecolor="white")
    ax1.axvline(c_icc, color="#D55E00", lw=2, label=f"mean ceiling = {c_icc:.3f}")
    ax1.axvline(own_aug, color="#000000", lw=1.5, ls="--", label=f"own R2 (aug) = {own_aug:.3f}")
    ax1.set_xlabel("per-context noise-ceiling R2 (ICC)")
    ax1.set_ylabel("contexts")
    ax1.set_title(f"Per-context ceiling spread (n={len(ceil_pc)})")
    ax1.legend(fontsize=8)

    # right: within vs total single-draw variance (each point a context)
    ax2.scatter(T, W, s=26, color="#0072B2", alpha=0.7, edgecolor="white", linewidth=0.4)
    lim = max(T.max(), W.max()) * 1.05
    ax2.plot([0, lim], [0, lim], color="#999999", lw=1, ls=":", label="W = T (ceiling 0)")
    ax2.set_xlabel("per-context total single-draw variance T")
    ax2.set_ylabel("per-context within-context variance W")
    ax2.set_title("Variance decomposition per context (lower W/T = higher ceiling)")
    ax2.set_xlim(0, lim)
    ax2.set_ylim(0, lim)
    ax2.legend(fontsize=8)

    fig.suptitle("Per-context noise-ceiling view (k=10 own resamples, layer-20, 42 position slots)")
    fig.tight_layout()
    savefig_paper(fig, "issue_952/noise_ceiling_percontext", dir=FIGDIR)
    plt.close(fig)


def main() -> None:
    report, pc = _load()
    fig_summary(report)
    fig_percontext(report, pc)
    print("figures written to figures/issue_952/noise_ceiling_{summary,percontext}.png")


if __name__ == "__main__":
    main()
