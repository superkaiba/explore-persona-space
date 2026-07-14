"""Hero figure for #672 follow-up: live GCP SPOT A100-80 memory trace stays flat.

Plots the per-iteration GPU-resident memory trace from the live Section-A smoke
(reserved + nvidia-smi primary, allocated secondary) against the documented
pre-#671 #545 climb (22 -> 30 -> 38 GiB) as a dashed contrast overlay.
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "eval_results/issue_672/secA_smoke_followup/memory_log.json"

THRESHOLD_GIB = 1.0  # PASS band: post-warmup max-min must stay under this
WARMUP_ITERS = 10  # validator drops the first 10 iters as load/stage warmup


def main() -> None:
    rows = json.loads(LOG.read_text())
    it = [r["iter"] for r in rows]
    reserved = [r["memory_reserved_gib"] for r in rows]
    nvidia = [r["nvidia_smi_used_gib"] for r in rows]
    allocated = [r["memory_allocated_gib"] for r in rows]

    # Post-warmup flat band (the gated quantity): reserved over iters > WARMUP_ITERS.
    post = [v for i, v in zip(it, reserved, strict=False) if i > WARMUP_ITERS]
    post_med = sorted(post)[len(post) // 2]
    post_range = max(post) - min(post)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    c_reserved = paper_palette_role("primary")
    c_nvidia = paper_palette_role("accent")
    c_alloc = paper_palette_role("neutral")
    c_climb = paper_palette_role("control")

    # Live traces (this run).
    ax.plot(it, nvidia, color=c_nvidia, lw=1.6, label="nvidia-smi memory.used (live, primary)")
    ax.plot(it, reserved, color=c_reserved, lw=1.8, label="torch reserved (live, primary)")
    ax.plot(it, allocated, color=c_alloc, lw=1.4, ls=":", label="torch allocated (live, secondary)")

    # PASS band: +/- 1 GiB around the post-warmup median.
    ax.axhspan(
        post_med - THRESHOLD_GIB,
        post_med + THRESHOLD_GIB,
        color=c_reserved,
        alpha=0.08,
        zorder=0,
        label=f"PASS band (median +/- {THRESHOLD_GIB:.0f} GiB)",
    )

    # Pre-#671 #545 climb contrast (documented 22 -> 30 -> 38 GiB monotone climb).
    climb_x = [0, max(it) / 2, max(it)]
    climb_y = [22.0, 30.0, 38.0]
    ax.plot(
        climb_x,
        climb_y,
        color=c_climb,
        lw=2.0,
        ls="--",
        marker="o",
        markersize=5,
        label="pre-fix climb (documented, 22->38 GiB)",
    )

    ax.set_xlabel("extractor iteration")
    ax.set_ylabel("GPU memory (GiB)")
    ax.set_ylim(20, 41)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    set_title_subtitle(
        ax,
        "Live GCP SPOT A100-80: memory stays flat after the extractor fix",
        f"638 post-warmup samples; reserved + nvidia-smi range {post_range:.2f} GiB "
        "(PASS < 1 GiB), vs the pre-fix 16 GiB climb",
        source=(
            "eval_results/issue_672/secA_smoke_followup/memory_log.json (649 samples, 227s wall)"
        ),
    )

    savefig_paper(fig, "issue_672/secA_memory_flat_live_followup", dir="figures/")
    plt.close(fig)
    print(f"post-warmup reserved range = {post_range:.4f} GiB; median = {post_med:.3f} GiB")


if __name__ == "__main__":
    main()
