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


LABELS = ["instruct\nsyco", "instruct\nhallu", "base\nsyco", "base\nhallu"]
ORDER = [f"{m}/{tr}" for m in ("instruct", "base") for tr in TRAITS]
_PE = "#4C72B0"  # prefix-end (pre-query)
_AC = "#DD8452"  # averaged context


def _draw_supervised(ax, t):
    """Supervised probe r (prefix-end vs averaged) with per-cell reliability ceiling."""
    x = np.arange(len(ORDER))
    w = 0.38
    pe = [t[k]["supervised_prefix_end_r"] for k in ORDER]
    ac = [t[k]["supervised_averaged_context_r"] for k in ORDER]
    pe_e = [
        _err(t[k]["supervised_prefix_end_ci95"], t[k]["supervised_prefix_end_r"]) for k in ORDER
    ]
    ac_e = [
        _err(t[k]["supervised_averaged_context_ci95"], t[k]["supervised_averaged_context_r"])
        for k in ORDER
    ]
    ceil = [t[k]["ceiling"] for k in ORDER]
    ax.bar(x - w / 2, pe, w, yerr=pe_e, capsize=3, label="prefix-end (pre-query)", color=_PE)
    ax.bar(x + w / 2, ac, w, yerr=ac_e, capsize=3, label="averaged context", color=_AC)
    for xi, c in zip(x, ceil):
        ax.hlines(c, xi - w, xi + w, color="#555555", linestyle="--", linewidth=1.3)
    ax.plot([], [], color="#555555", linestyle="--", label="reliability ceiling")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("supervised probe  r  (state → per-prefix judge mean)")
    ax.set_title("Supervised read of average behavioral disposition vs its reliability ceiling")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.05, 1.0)
    ax.legend(fontsize=8, loc="upper right")


def _draw_rb(ax, t):
    """Raw persona-vector (r_B) projection (signed), prefix-end vs averaged."""
    x = np.arange(len(ORDER))
    w = 0.38
    rpe = [t[k]["raw_rb_prefix_end_r"] for k in ORDER]
    rac = [t[k]["raw_rb_averaged_context_r"] for k in ORDER]
    rpe_e = [_err(t[k]["raw_rb_prefix_end_ci95"], t[k]["raw_rb_prefix_end_r"]) for k in ORDER]
    rac_e = [
        _err(t[k]["raw_rb_averaged_context_ci95"], t[k]["raw_rb_averaged_context_r"]) for k in ORDER
    ]
    ax.bar(x - w / 2, rpe, w, yerr=rpe_e, capsize=3, label="prefix-end (pre-query)", color=_PE)
    ax.bar(x + w / 2, rac, w, yerr=rac_e, capsize=3, label="averaged context", color=_AC)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("raw persona-vector (r_B) projection  r")
    ax.set_title("Unsupervised r_B projection (no fitted map)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.6, 0.85)
    ax.legend(fontsize=8, loc="upper right")


def plot(reframe):
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()
    t = reframe["table"]
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # Two standalone single-panel figures — one per result in the writeup.
    fig1, ax1 = plt.subplots(figsize=(7, 4.6))
    _draw_supervised(ax1, t)
    fig1.tight_layout()
    for ext in ("png", "pdf"):
        fig1.savefig(FIGDIR / f"base_prequery_supervised.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 4.6))
    _draw_rb(ax2, t)
    fig2.tight_layout()
    for ext in ("png", "pdf"):
        fig2.savefig(FIGDIR / f"base_prequery_rb.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Combined figure retained (referenced by the two direct-vs-averaged summary docs).
    figc, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    _draw_supervised(axL, t)
    _draw_rb(axR, t)
    figc.suptitle(
        "#1092: instruction tuning installs a pre-query disposition the base model lacks (layer 14)",
        fontsize=11,
    )
    figc.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        figc.savefig(FIGDIR / f"base_prequery_reframe.{ext}", dpi=150, bbox_inches="tight")
    plt.close(figc)

    (FIGDIR / "base_prequery_reframe.meta.json").write_text(
        json.dumps(
            {
                "script": "scripts/issue1092_base_prequery_reframe.py",
                "sources": reframe["source_artifacts"],
                "layer": 14,
                "figures": [
                    "base_prequery_supervised.png",
                    "base_prequery_rb.png",
                    "base_prequery_reframe.png",
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    rf = build()
    plot(rf)
    print("wrote", OUT / "base_prequery_reframe.json")
    for name in (
        "base_prequery_supervised.png",
        "base_prequery_rb.png",
        "base_prequery_reframe.png",
    ):
        print("wrote", FIGDIR / name)
