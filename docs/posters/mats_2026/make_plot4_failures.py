"""MATS 2026 poster figure 4: retrieval-metric ladder + residual-failure margins.

Replaces the raw-euclidean failure-category figure: under the fixed retrieval
metric (whitened cosine + CSLS K=10 + 5-draw-averaged targets — the committed
sampling-noise-free "deterministic answer" condition; no greedy arm exists in
#2202) only 11 residual failures remain of 1,988 covered rows, so the
per-category bar panel's premise dissolves. Two panels, every number from
committed eval_results/issue_2202 JSONs:

- panel (a) METRIC LADDER: rank-1 retrieval accuracy across the four rungs
  raw euclidean (0.8160, pool 9,941; repro_gate.json) -> whitened cosine
  (0.9535, pool 9,941; metric_zoo/summary.json banked_baselines) -> + CSLS
  K=10 (0.9762, pool 9,941; metric_zoo new_conventions_ranked) -> + 5-draw-
  averaged targets (ridge 0.9945 on the 1,988 resample-covered rows, pool
  9,941; avgtgt_completion/summary.json matrix). The dashed line is the
  convention-matched fresh-draw ceiling 0.9731 (metric_zoo ceilings). The 6
  non-ridge architectures at the final rung render as small open markers
  (0.9909-0.9945 spread).
- panel (b) RESIDUAL FAILURES: per-context retrieval margins (true target's
  score minus best competitor's, CSLS K=10 whitened cosine, draw-averaged
  targets) over the 1,988 covered rows, log-count histogram
  (residual_read/percontext_ranks_margins.npz field
  margin_csls_k10_whitencos_avg). The 11 failures lose by a median 0.025
  while the 1,977 successes win by a median 0.175 (~7x); pairwise AUC
  0.9999969 (residual_read/summary.json part2_differentiation).

Writes ``docs/posters/mats_2026/figures/plot4_failures.{png,pdf,meta.json}``
plus the audit sidecar ``plot4_failures_data.json`` with every plotted number.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

EV = REPO / "eval_results" / "issue_2202"
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"

RUNG_LABELS = [
    "raw\neuclidean",
    "whitened\ncosine",
    "+ CSLS\n(K = 10)",
    "+ 5-draw-avg.\ntargets",
]
FINAL_CONVENTION = "csls_k10_whitencos"


def load_ladder() -> dict:
    """The four rung values + matched ceiling + per-map final-rung spread."""
    rg = json.loads((EV / "repro_gate.json").read_text())
    rung1 = rg["metrics"]["euclidean"]["recomputed"]["acc_at_k"]["1"]

    mz = json.loads((EV / "metric_zoo" / "summary.json").read_text())
    rung2 = mz["banked_baselines"]["acc1_table"]["whiten_cos"]["acc1"]
    rung3 = next(c["acc1"] for c in mz["new_conventions_ranked"] if c["name"] == FINAL_CONVENTION)
    ceiling = mz["ceilings"][f"ceiling_{FINAL_CONVENTION}"]["ceiling"]["acc1_ceiling"]

    av = json.loads((EV / "avgtgt_completion" / "summary.json").read_text())
    per_map = {
        name: cell[FINAL_CONVENTION]["avg"]["acc_at_k"]["1"] for name, cell in av["matrix"].items()
    }
    return {
        "rungs": [rung1, rung2, rung3, per_map["ridge"]],
        "ceiling": ceiling,
        "per_map": per_map,
        "n_covered": av["n_covered"],
        "n_pool": av["n_pool"],
    }


def main() -> None:
    set_paper_style("iclr", font_scale=1.5)
    lad = load_ladder()

    z = np.load(EV / "residual_read" / "percontext_ranks_margins.npz")
    margins = z["margin_csls_k10_whitencos_avg"]
    ranks = z["rank_csls_k10_whitencos_avg"]
    fail = ranks > 1.0
    assert int(fail.sum()) == int((margins < 0).sum()), "margin sign vs rank mismatch"

    diff = json.loads((EV / "residual_read" / "summary.json").read_text())["part2_differentiation"][
        FINAL_CONVENTION
    ]["avg"]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )

    # (a) metric ladder, ridge map; other architectures at the final rung
    xs = np.arange(4)
    ax_a.plot(
        xs,
        lad["rungs"],
        marker="o",
        markersize=7,
        color=paper_color("instruct"),
        lw=1.6,
        label="ridge map",
        zorder=3,
    )
    others = [v for k, v in lad["per_map"].items() if k != "ridge"]
    ax_a.scatter(
        np.full(len(others), 3.22),
        others,
        s=22,
        facecolors="none",
        edgecolors=paper_color("null"),
        linewidths=1.2,
        label="6 other architectures",
        zorder=2,
    )
    ax_a.axhline(
        lad["ceiling"],
        color=paper_color("reference"),
        ls="--",
        lw=1.1,
        label="fresh-draw ceiling (matched)",
    )
    ax_a.set_xticks(xs, RUNG_LABELS)
    ax_a.set_xlim(-0.35, 3.55)
    ax_a.set_ylim(0.78, 1.005)
    ax_a.set_ylabel("rank-1 retrieval accuracy")
    ax_a.legend(loc="lower right", handlelength=1.4)

    # (b) per-context retrieval margins under the fixed metric
    bins = np.linspace(-0.16, 0.46, 42)
    ax_b.hist(
        margins[~fail],
        bins=bins,
        color=paper_color("null"),
        label=f"successes (n = {int((~fail).sum()):,})",
    )
    ax_b.hist(
        margins[fail],
        bins=bins,
        color=paper_color("instruct"),
        label=f"residual failures (n = {int(fail.sum())})",
    )
    ax_b.axvline(0.0, color=paper_color("reference"), lw=1.0)
    ax_b.set_yscale("log")
    ax_b.set_ylim(top=1400)
    ax_b.set_xlabel("retrieval margin (true $-$ best competitor)")
    ax_b.set_ylabel("contexts (log)")
    ax_b.legend(loc="upper right", handlelength=1.0)

    savefig_paper(fig, "plot4_failures", dir=OUT_DIR)
    plt.close(fig)

    sidecar = {
        "panel_a": {
            "rung_labels": RUNG_LABELS,
            "rungs_acc1": lad["rungs"],
            "rung_pools": ["9,941 full pool"] * 3 + ["1,988 resample-covered rows (pool 9,941)"],
            "fresh_draw_ceiling_matched": lad["ceiling"],
            "per_map_final_rung": lad["per_map"],
            "sources": {
                "raw_euclidean": "eval_results/issue_2202/repro_gate.json"
                " .metrics.euclidean.recomputed.acc_at_k.1",
                "whiten_cos": "eval_results/issue_2202/metric_zoo/summary.json"
                " .banked_baselines.acc1_table.whiten_cos.acc1",
                "csls_k10_whitencos": "eval_results/issue_2202/metric_zoo/summary.json"
                " .new_conventions_ranked[name=csls_k10_whitencos].acc1",
                "avg_targets": "eval_results/issue_2202/avgtgt_completion/summary.json"
                " .matrix[map].csls_k10_whitencos.avg.acc_at_k.1",
                "ceiling": "eval_results/issue_2202/metric_zoo/summary.json"
                " .ceilings.ceiling_csls_k10_whitencos.ceiling.acc1_ceiling",
            },
        },
        "panel_b": {
            "n_rows": len(margins),
            "n_failures": int(fail.sum()),
            "margin_stats": diff,
            "fail_margins_sorted": np.sort(margins[fail]).tolist(),
            "sources": {
                "margins": "eval_results/issue_2202/residual_read/percontext_ranks_margins.npz"
                " field margin_csls_k10_whitencos_avg (fail = rank_csls_k10_whitencos_avg > 1)",
                "stats": "eval_results/issue_2202/residual_read/summary.json"
                " .part2_differentiation.csls_k10_whitencos.avg",
            },
        },
    }
    (OUT_DIR / "plot4_failures_data.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {OUT_DIR}/plot4_failures.{{png,pdf,meta.json}} + plot4_failures_data.json")
    print("ladder:", [f"{v:.4f}" for v in lad["rungs"]], "ceiling:", f"{lad['ceiling']:.4f}")
    print("per-map final rung:", {k: f"{v:.4f}" for k, v in lad["per_map"].items()})
    print(
        f"margins: {int(fail.sum())} failures median {np.median(margins[fail]):+.4f}, "
        f"{int((~fail).sum())} successes median {np.median(margins[~fail]):+.4f}, "
        f"pairwise AUC {diff['pairwise_auc']:.7f}"
    )


if __name__ == "__main__":
    main()
