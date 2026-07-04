"""Issue #958 round-4 figure: λ-clamped long-panel turn-1 transfer at turns 5-8.

Renders figures/issue_958/long_turn1_transfer_lclamp.{png,pdf,meta.json} from
the committed eval_results/issue_958/long_k1_transfer{,_lclamp}.json aggregates
and the percell long_1to{k}_lclamp.npz / long_own_k{k}.npz cells. No new
computation — a pure re-plot of persisted artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import issue958_common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

OUT = Path("eval_results/issue_958")
RO = [C.block_to_row(b) for b in C.READOUT_BLOCKS]
KS = [5, 6, 7, 8]
DUP_TEST_CI = [113, 290, 317, 372, 491, 503, 526, 591]


def main() -> None:
    set_paper_style("blog")
    res_gcv = json.loads((OUT / "long_k1_transfer.json").read_text())
    res_cl = json.loads((OUT / "long_k1_transfer_lclamp.json").read_text())
    own = {k: np.load(OUT / "percell" / f"long_own_k{k}.npz") for k in KS}
    test_idx = own[5]["test_idx"]
    idx = np.random.default_rng(C.BOOTSTRAP_SEED).integers(
        0, len(test_idx), size=(C.BOOTSTRAP_DRAWS, len(test_idx))
    )

    def boot(sse: np.ndarray, null: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                1.0 - sse[r][idx].sum(1) / np.clip(null[r][idx].sum(1), 1e-30, None)
                for r in range(sse.shape[0])
            ]
        ).mean(0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    cols = paper_palette(4)
    ax = axes[0]
    xs = np.arange(len(KS))
    w = 0.2
    own_p = [float(np.mean([own[k]["skill"][r] for r in RO])) for k in KS]
    own_ci = []
    for k in KS:
        ob = boot(own[k]["sse_unit"][RO], own[k]["null_sse_unit"][RO])
        own_ci.append([float(np.quantile(ob, q)) for q in (0.025, 0.975)])
    gcv_p = [res_gcv["cells"][f"long_1to{k}"]["transfer_skill"] for k in KS]
    gcv_ci = [res_gcv["cells"][f"long_1to{k}"]["transfer_skill_ci95"] for k in KS]
    cl_p = [res_cl["cells"][f"long_1to{k}"]["transfer_skill"] for k in KS]
    cl_ci = [res_cl["cells"][f"long_1to{k}"]["transfer_skill_ci95"] for k in KS]
    rec_p = [res_cl["cells"][f"long_1to{k}"]["recalibrated_transfer_skill"] for k in KS]
    rec_ci = [res_cl["cells"][f"long_1to{k}"]["recalibrated_transfer_skill_ci95"] for k in KS]

    def err(p: list, ci: list) -> np.ndarray:
        return np.abs(np.array(ci).T - np.array(p))

    ax.bar(xs - 1.5 * w, own_p, w, yerr=err(own_p, own_ci), color=cols[0], label="own-turn map")
    ax.bar(
        xs - 0.5 * w,
        gcv_p,
        w,
        yerr=err(gcv_p, gcv_ci),
        color=cols[1],
        label="turn-1 map, GCV λ≈5 (as fitted)",
    )
    ax.bar(
        xs + 0.5 * w,
        cl_p,
        w,
        yerr=err(cl_p, cl_ci),
        color=cols[2],
        label="turn-1 map, λ clamped to 1,000",
    )
    ax.bar(
        xs + 1.5 * w,
        rec_p,
        w,
        yerr=err(rec_p, rec_ci),
        color=cols[3],
        label="λ-clamped + target-turn moments",
    )
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xticks(xs, [f"turn {k}" for k in KS])
    ax.set_ylim(-0.58, 0.46)
    ax.set_ylabel("held-out skill (6-block mean)")
    ax.set_xlabel("evaluation turn (long panel, 60 test conversations)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax2 = axes[1]
    k = 5
    po = 1 - own[k]["sse_unit"][RO].sum(0) / np.clip(
        own[k]["null_sse_unit"][RO].sum(0), 1e-30, None
    )
    pz = np.load(OUT / "percell" / f"long_1to{k}_lclamp.npz")
    pt = 1 - pz["sse_unit"].sum(0) / np.clip(pz["null_sse_unit"].sum(0), 1e-30, None)
    po_c, pt_c = np.clip(po, -1, None), np.clip(pt, -1, None)
    dup = np.isin(test_idx, DUP_TEST_CI)
    ax2.scatter(
        po_c[~dup], pt_c[~dup], s=24, color=cols[2], alpha=0.8, label="unique first message"
    )
    ax2.scatter(
        po_c[dup],
        pt_c[dup],
        s=34,
        facecolors="none",
        edgecolors=cols[1],
        linewidths=1.2,
        label="duplicate first message",
    )
    lim = [-1.05, 1.0]
    ax2.plot(lim, lim, color="0.6", lw=0.8, ls="--")
    lows = np.argsort(pt_c)[:2]
    for j, i in enumerate(lows):
        ax2.text(
            po_c[i] + 0.03,
            pt_c[i] + (0.05 if j else -0.09),
            f"conv {int(test_idx[i])}",
            fontsize=7,
            va="bottom",
        )
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("own turn-5 map skill per conversation (clipped at -1)")
    ax2.set_ylabel("λ-clamped turn-1 map skill at turn 5 (clipped at -1)")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    savefig_paper(fig, "long_turn1_transfer_lclamp", dir="figures/issue_958")
    print("wrote figures/issue_958/long_turn1_transfer_lclamp.png")


if __name__ == "__main__":
    main()
