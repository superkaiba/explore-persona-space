"""Reframe the #1092 prefix-end monitoring 'base reads collapse' aside.

Pure re-read (0 GPU, 0 download) of two already-local artifacts:
  eval_results/issue_1092/inline_prefixend_monitoring/results.json
  eval_results/issue_1092/inline_prefixend_monitoring/readout_constructions.json

It splits the one-line base-model collapse into its two distinct phenomena by
reporting, per cell x trait, the supervised-probe r alongside the per-prefix
target reliability ceiling (so a low r against a low ceiling reads differently
from a low r against a high one), and by surfacing the base-cell RAW r_B
projection that was computed in readout_constructions.json but never written up.

Emits a table JSON + one figure. No model forwards, no fits, no new data.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "eval_results/issue_1092/inline_prefixend_monitoring"
OUT = REPO / "eval_results/issue_1092/inline_base_prequery_reframe"
FIGDIR = REPO / "figures/issue_1092"

CELLS = {"cell_inst_own": "instruct", "cell_pre_own": "base"}
TRAITS = ("sycophancy", "hallucination")
LAYER = "14"


def _r(x):
    """Pull a scalar r from either a bare float or a {'r': ...} block."""
    if isinstance(x, dict):
        return x.get("r")
    return x


def build():
    results = json.loads((SRC / "results.json").read_text())
    rc = json.loads((SRC / "readout_constructions.json").read_text())

    table = {}
    for cell, model in CELLS.items():
        # results.json carries reliability + ceiling per trait block
        rel = {}
        for blk in results["cells"][cell][LAYER]:
            rel[blk["trait"]] = {
                "n_prefixes": blk["n_prefixes"],
                "n_rows": blk["n_rows"],
                "target_reliability_per_prefix_mean": blk["target_reliability_per_prefix_mean"],
                "target_split_half_r_aligned": blk["target_split_half_r_aligned"],
                "ceiling": blk["monitoring_r_ceiling_from_reliability"],
            }
        for trait in TRAITS:
            sa = rc["supervised_anchor"][cell][trait]
            raw = rc["constructions"]["raw_rb_projection"][cell][trait]
            mm = rc["constructions"]["map_mediated"][cell][trait]
            ceil = rel[trait]["ceiling"]
            sup_pe = _r(sa["prefix_end"])
            sup_ac = _r(sa["averaged_context"])
            table[f"{model}/{trait}"] = {
                "model": model,
                "trait": trait,
                "cell": cell,
                "n_prefixes": rel[trait]["n_prefixes"],
                "n_rows": rel[trait]["n_rows"],
                "target_reliability": rel[trait]["target_reliability_per_prefix_mean"],
                "target_split_half_r_aligned": rel[trait]["target_split_half_r_aligned"],
                "ceiling": ceil,
                # supervised ridge probe -> per-prefix judge mean
                "supervised_prefix_end_r": sup_pe,
                "supervised_averaged_context_r": sup_ac,
                "supervised_prefix_end_ci95": sa.get("prefix_end_ci95"),
                "supervised_averaged_context_ci95": sa.get("averaged_context_ci95"),
                # ceiling-normalized (fraction of the reachable signal captured)
                "supervised_prefix_end_frac_ceiling": (sup_pe / ceil) if ceil else None,
                "supervised_averaged_context_frac_ceiling": (sup_ac / ceil) if ceil else None,
                # raw persona-vector (r_B) projection -> per-prefix judge mean
                "raw_rb_prefix_end_r": _r(raw["prefix_end"]),
                "raw_rb_averaged_context_r": _r(raw["averaged_context"]),
                "raw_rb_prefix_end_ci95": raw["prefix_end"].get("ci95"),
                "raw_rb_averaged_context_ci95": raw["averaged_context"].get("ci95"),
                # map-mediated (transport state through the fitted map, then read)
                "map_mediated_prefix_end_r": _r(mm["prefix_end"]),
                "map_mediated_averaged_context_r": _r(mm["averaged_context"]),
            }

    reframe = {
        "read": "#1092 base-model pre-query-disposition reframe (inline free-analysis, 0 GPU)",
        "source_artifacts": [
            "eval_results/issue_1092/inline_prefixend_monitoring/results.json",
            "eval_results/issue_1092/inline_prefixend_monitoring/readout_constructions.json",
        ],
        "layer": 14,
        "note": (
            "Splits the one-line 'base reads collapse' aside into two phenomena: "
            "(1) base sycophancy has near-zero between-prefix target signal "
            "(reliability 0.078, ceiling 0.278) so there is little to read; "
            "(2) base hallucination has real signal (averaged read 0.62 ~= 0.88 of "
            "ceiling 0.709) that only forms POST-query -- the pre-query prefix-end "
            "state does not carry it (supervised 0.05, raw r_B proj 0.02, both CIs "
            "straddle 0), the clean mirror of the instruct model whose prefix-end "
            "state holds the disposition before the query (r_B proj 0.665 sycophancy, "
            "-0.43 hallucination)."
        ),
        "table": table,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "base_prequery_reframe.json").write_text(json.dumps(reframe, indent=2))
    return reframe


def _err(ci, center):
    if not ci:
        return 0.0
    return max(abs(center - ci[0]), abs(ci[1] - center))


def plot(reframe):
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    t = reframe["table"]
    order = [f"{m}/{tr}" for m in ("instruct", "base") for tr in TRAITS]
    labels = ["instruct\nsyco", "instruct\nhallu", "base\nsyco", "base\nhallu"]
    x = np.arange(len(order))
    w = 0.38

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel A: supervised probe r, prefix-end vs averaged, with ceiling markers.
    pe = [t[k]["supervised_prefix_end_r"] for k in order]
    ac = [t[k]["supervised_averaged_context_r"] for k in order]
    pe_e = [
        _err(t[k]["supervised_prefix_end_ci95"], t[k]["supervised_prefix_end_r"]) for k in order
    ]
    ac_e = [
        _err(t[k]["supervised_averaged_context_ci95"], t[k]["supervised_averaged_context_r"])
        for k in order
    ]
    ceil = [t[k]["ceiling"] for k in order]
    axL.bar(x - w / 2, pe, w, yerr=pe_e, capsize=3, label="prefix-end (pre-query)", color="#4C72B0")
    axL.bar(x + w / 2, ac, w, yerr=ac_e, capsize=3, label="averaged context", color="#DD8452")
    for xi, c in zip(x, ceil):
        axL.hlines(c, xi - w, xi + w, color="#555555", linestyle="--", linewidth=1.3)
    axL.plot([], [], color="#555555", linestyle="--", label="reliability ceiling")
    axL.set_xticks(x)
    axL.set_xticklabels(labels)
    axL.set_ylabel("supervised probe  r  (state → per-prefix judge mean)")
    axL.set_title("Supervised read vs its reliability ceiling")
    axL.axhline(0, color="black", linewidth=0.8)
    axL.set_ylim(-0.05, 1.0)
    axL.legend(fontsize=8, loc="upper right")

    # Panel B: raw r_B projection (signed), prefix-end vs averaged.
    rpe = [t[k]["raw_rb_prefix_end_r"] for k in order]
    rac = [t[k]["raw_rb_averaged_context_r"] for k in order]
    rpe_e = [_err(t[k]["raw_rb_prefix_end_ci95"], t[k]["raw_rb_prefix_end_r"]) for k in order]
    rac_e = [
        _err(t[k]["raw_rb_averaged_context_ci95"], t[k]["raw_rb_averaged_context_r"]) for k in order
    ]
    axR.bar(
        x - w / 2, rpe, w, yerr=rpe_e, capsize=3, label="prefix-end (pre-query)", color="#4C72B0"
    )
    axR.bar(x + w / 2, rac, w, yerr=rac_e, capsize=3, label="averaged context", color="#DD8452")
    axR.set_xticks(x)
    axR.set_xticklabels(labels)
    axR.set_ylabel("raw persona-vector (r_B) projection  r")
    axR.set_title("Unsupervised r_B projection (no fitted map)")
    axR.axhline(0, color="black", linewidth=0.8)
    axR.set_ylim(-0.6, 0.85)
    axR.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "#1092: instruction tuning installs a pre-query disposition the base model lacks (layer 14)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"base_prequery_reframe.{ext}", dpi=150, bbox_inches="tight")
    (FIGDIR / "base_prequery_reframe.meta.json").write_text(
        json.dumps(
            {
                "script": "scripts/issue1092_base_prequery_reframe.py",
                "sources": reframe["source_artifacts"],
                "layer": 14,
            },
            indent=2,
        )
    )
    plt.close(fig)


if __name__ == "__main__":
    rf = build()
    plot(rf)
    print("wrote", OUT / "base_prequery_reframe.json")
    print("wrote", FIGDIR / "base_prequery_reframe.png")
