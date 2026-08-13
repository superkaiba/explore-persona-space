#!/usr/bin/env python
"""Framing-5 (`indirect` reported speech) figures for issue #2054.

Builds, from the framing-5 extension round's artifacts:
  (i)  f5_ceilings          — indirect within-cell ceilings (both arms) beside the
                              four banked framings' on-policy ceilings;
  (ii) f5_ladder_rungs      — 9-rung transfer ladder profiles for the two
                              class-valid pair classes (cross-character,
                              cross-model), per arm, median + IQR;
  (iii) f5_ladder_rungs_units — per-unit spaghetti companion of (ii).

Inputs:
  - HF-downloaded indirect fit JSONs   (issue2054_lattice/fits/*indirect*.json)
  - HF-downloaded indirect rung JSONs  (issue2054_lattice/ladder/rung_1_*indirect*.json)
  - committed pool_specialize digest   (banked per-cell ceilings for the four
    shipped framings; eval_results/issue_2054/pool_specialize/digest.json)

Also writes eval_results/issue_2054/f5_indirect/digest.json (per-cell ceilings,
per-unit ladder rows, summaries) so the figures are reproducible from a
committed artifact.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parents[1]
HF_LOCAL = Path("/tmp/issue2054_f5_hf/issue2054_lattice")
POOL_DIGEST = REPO / "eval_results/issue_2054/pool_specialize/digest.json"
OUT_FIG = "issue_2054/f5_indirect"
OUT_DIGEST = REPO / "eval_results/issue_2054/f5_indirect/digest.json"

RUNGS = [
    "1_direct",
    "2_ctx_offset",
    "3_ans_offset",
    "4_bias_refit",
    "5_global_scale",
    "6_rotation",
    "7_ctx_reparam",
    "8_ans_reparam",
    "9_full_AMB",
]
RUNG_LABEL = {
    "1_direct": "direct",
    "2_ctx_offset": "+ context offset",
    "3_ans_offset": "+ answer offset",
    "4_bias_refit": "+ bias refit",
    "5_global_scale": "+ scale",
    "6_rotation": "+ rotation",
    "7_ctx_reparam": "context re-map",
    "8_ans_reparam": "answer re-map",
    "9_full_AMB": "full refit",
}
FRAMING_LABEL = {
    "attrib_quoted": "attributed quote",
    "bare_label": "bare label",
    "chat": "chat template",
    "bare_text": "bare text",
    "indirect": "indirect report",
}
FRAMING_ORDER = ["attrib_quoted", "bare_label", "chat", "bare_text", "indirect"]
MODEL_LABEL = {"qwen2.5-7b": "base", "qwen2.5-7b-instruct": "instruct"}
CLASS_LABEL = {
    "cross_character": "cross-character",
    "cross_model": "cross-model",
}

BLUE, ORANGE = paper_palette_blog(2)


def parse_cell(cell: str) -> dict[str, str]:
    variant, condition, framing, model = cell.rsplit("__", 3)
    return {"variant": variant, "condition": condition, "framing": framing, "model": model}


def load_indirect_ceilings() -> list[dict]:
    rows = []
    for f in sorted((HF_LOCAL / "fits").glob("*indirect*.json")):
        d = json.loads(f.read_text())
        meta = parse_cell(d["cell"])
        for arm in ("context", "prefix"):
            pooled = d["arm_reports"][arm]["pooled"]
            rows.append(
                {
                    "cell": d["cell"],
                    "framing": "indirect",
                    "model": meta["model"],
                    "variant": meta["variant"],
                    "arm": arm,
                    "ceiling_r2": pooled["r2_ambient_mean"],
                    "null_r2_p95": pooled["null_r2_pooled_p95"],
                    "identity_bias_r2": pooled["r2_identity_bias_mean"],
                }
            )
    return rows


def load_banked_on_policy() -> list[dict]:
    digest = json.loads(POOL_DIGEST.read_text())
    rows = []
    for rec in digest["per_cell"]:
        if "__on_policy__" not in rec["cell"]:
            continue
        meta = parse_cell(rec["cell"])
        rows.append(
            {
                "cell": rec["cell"],
                "framing": meta["framing"],
                "model": meta["model"],
                "variant": meta["variant"],
                "arm": rec["arm"],
                "ceiling_r2": rec["ceiling_r2"],
            }
        )
    return rows


def load_ladder_units() -> list[dict]:
    rows = []
    for f in sorted((HF_LOCAL / "ladder").glob("rung_1_*indirect*.json")):
        d = json.loads(f.read_text())
        rep = d["arm_report"]
        src, tgt = parse_cell(d["source"]), parse_cell(d["target"])
        if src["model"] == tgt["model"] and src["variant"] != tgt["variant"]:
            pair_class = "cross_character"
        elif src["variant"] == tgt["variant"] and src["model"] != tgt["model"]:
            pair_class = "cross_model"
        else:  # pragma: no cover - class-valid classes only were run
            raise ValueError(f"unexpected pair class for {f.name}")
        row = {
            "source": d["source"],
            "target": d["target"],
            "arm": rep["arm"],
            "pair_class": pair_class,
            "n_intersection": rep["n_intersection"],
            "target_ceiling": rep["target_ceiling"],
            "rung_r2_mean": {r: rep["pooled"][r]["r2_transfer_mean"] for r in RUNGS},
            "rung_ratio_mean": {r: rep["pooled"][r]["ratio_mean"] for r in RUNGS},
        }
        rows.append(row)
    return rows


def q(vals: list[float], p: float) -> float:
    return float(np.quantile(np.asarray(vals, dtype=float), p))


def fig_ceilings(indirect: list[dict], banked: list[dict]) -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    rows = banked + indirect
    rng = np.random.default_rng(42)
    for gi, framing in enumerate(FRAMING_ORDER):
        for arm, color, dx in (("context", BLUE, -0.18), ("prefix", ORANGE, 0.18)):
            sub = [r for r in rows if r["framing"] == framing and r["arm"] == arm]
            if not sub:
                continue
            xs = gi + dx + rng.uniform(-0.08, 0.08, len(sub))
            ys = [r["ceiling_r2"] for r in sub]
            filled = [r["model"] == "qwen2.5-7b-instruct" for r in sub]
            for x, y, is_instr in zip(xs, ys, filled):
                if is_instr:
                    ax.scatter([x], [y], s=42, color=color, alpha=0.85, zorder=3)
                else:
                    ax.scatter(
                        [x],
                        [y],
                        s=42,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.3,
                        alpha=0.9,
                        zorder=3,
                    )
            med = statistics.median(ys)
            ax.hlines(med, gi + dx - 0.16, gi + dx + 0.16, color=color, linewidth=2.6, zorder=4)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
    ax.set_xticks(range(len(FRAMING_ORDER)))
    ax.set_xticklabels([FRAMING_LABEL[f] for f in FRAMING_ORDER])
    ax.set_ylabel("within-cell held-out R²")
    # legend proxies
    ax.scatter([], [], s=42, color=BLUE, label="context arm, instruct")
    ax.scatter(
        [], [], s=42, facecolors="none", edgecolors=BLUE, linewidths=1.3, label="context arm, base"
    )
    ax.scatter([], [], s=42, color=ORANGE, label="prefix arm, instruct")
    ax.scatter(
        [], [], s=42, facecolors="none", edgecolors=ORANGE, linewidths=1.3, label="prefix arm, base"
    )
    ax.legend(loc="upper right", ncol=2)
    set_title_subtitle(
        ax,
        "Indirect-report ceilings land inside the banked on-policy range",
        "one point per on-policy cell; horizontal bars mark per-framing, per-arm medians",
    )
    savefig_paper(fig, f"{OUT_FIG}/f5_ceilings", dir="figures/")
    plt.close(fig)


def fig_ladder_summary(units: list[dict]) -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, arm in zip(axes, ("context", "prefix")):
        for pair_class, color in (("cross_character", BLUE), ("cross_model", ORANGE)):
            sub = [u for u in units if u["arm"] == arm and u["pair_class"] == pair_class]
            med = [statistics.median([u["rung_r2_mean"][r] for u in sub]) for r in RUNGS]
            lo = [q([u["rung_r2_mean"][r] for u in sub], 0.25) for r in RUNGS]
            hi = [q([u["rung_r2_mean"][r] for u in sub], 0.75) for r in RUNGS]
            x = np.arange(len(RUNGS))
            ax.plot(
                x,
                med,
                marker="o",
                color=color,
                label=f"{CLASS_LABEL[pair_class]} (n={len(sub)})",
            )
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, linewidth=0)
            ceil_med = statistics.median([u["target_ceiling"] for u in sub])
            ax.hlines(
                ceil_med,
                -0.3,
                len(RUNGS) - 0.7,
                color=color,
                linestyle="--",
                linewidth=1.1,
                alpha=0.7,
            )
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
        ax.set_xticks(np.arange(len(RUNGS)))
        ax.set_xticklabels([RUNG_LABEL[r] for r in RUNGS], rotation=30, ha="right")
        ax.set_title(f"{arm} arm", loc="left", pad=6, fontsize=12)
        if arm == "context":
            ax.set_ylabel("transfer held-out R², mean over folds")
            ax.legend(loc="lower right")
    fig.suptitle(
        "Indirect-report transfer ladder by pair class "
        "(lines = medians, bands = interquartile range; dashed = median target ceiling)",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, f"{OUT_FIG}/f5_ladder_rungs", dir="figures/")
    plt.close(fig)


def fig_ladder_units(units: list[dict]) -> None:
    set_paper_style("blog")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, arm in zip(axes, ("context", "prefix")):
        for pair_class, color in (("cross_character", BLUE), ("cross_model", ORANGE)):
            sub = [u for u in units if u["arm"] == arm and u["pair_class"] == pair_class]
            x = np.arange(len(RUNGS))
            for u in sub:
                ax.plot(
                    x,
                    [u["rung_r2_mean"][r] for r in RUNGS],
                    color=color,
                    alpha=0.35,
                    linewidth=1.0,
                )
            med = [statistics.median([u["rung_r2_mean"][r] for u in sub]) for r in RUNGS]
            ax.plot(
                x,
                med,
                color=color,
                linewidth=2.8,
                label=f"{CLASS_LABEL[pair_class]} median (n={len(sub)})",
            )
        ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.0)
        ax.set_xticks(np.arange(len(RUNGS)))
        ax.set_xticklabels([RUNG_LABEL[r] for r in RUNGS], rotation=30, ha="right")
        ax.set_title(f"{arm} arm", loc="left", pad=6, fontsize=12)
        if arm == "context":
            ax.set_ylabel("transfer held-out R², mean over folds")
            ax.legend(loc="lower right")
    fig.suptitle(
        "Indirect-report transfer ladder, per pair "
        "(one thin line per ordered pair; thick lines = class medians)",
        x=0.02,
        ha="left",
        fontsize=11,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    savefig_paper(fig, f"{OUT_FIG}/f5_ladder_rungs_units", dir="figures/")
    plt.close(fig)


def main() -> None:
    indirect = load_indirect_ceilings()
    banked = load_banked_on_policy()
    units = load_ladder_units()
    assert len(indirect) == 20, len(indirect)  # 10 cells x 2 arms
    assert len(banked) == 48, len(banked)  # 24 on-policy cells x 2 arms
    assert len(units) == 68, len(units)

    # summaries
    def ceil_summary(rows: list[dict], arm: str) -> dict:
        vals = sorted(r["ceiling_r2"] for r in rows if r["arm"] == arm)
        return {
            "n": len(vals),
            "min": vals[0],
            "max": vals[-1],
            "median": statistics.median(vals),
        }

    ladder_summary: dict = {}
    for arm in ("context", "prefix"):
        for pc in ("cross_character", "cross_model"):
            sub = [u for u in units if u["arm"] == arm and u["pair_class"] == pc]
            ladder_summary[f"{pc}__{arm}"] = {
                "n_pairs": len(sub),
                "median_target_ceiling": statistics.median([u["target_ceiling"] for u in sub]),
                "rung_r2_median": {
                    r: statistics.median([u["rung_r2_mean"][r] for u in sub]) for r in RUNGS
                },
                "rung_ratio_median": {
                    r: statistics.median([u["rung_ratio_mean"][r] for u in sub]) for r in RUNGS
                },
                "n_intersection_range": [
                    min(u["n_intersection"] for u in sub),
                    max(u["n_intersection"] for u in sub),
                ],
            }

    digest = {
        "metadata": {
            "script": "scripts/issue2054_f5_indirect_figs.py",
            "sources": {
                "indirect_fits": "issue2054_lattice/fits/*indirect*.json (HF data repo)",
                "indirect_ladder": "issue2054_lattice/ladder/rung_1_*indirect*.json (HF data repo)",
                "banked_ceilings": "eval_results/issue_2054/pool_specialize/digest.json per_cell ceiling_r2",
            },
            "provenance": {
                "substrate": "on-policy generations only for the indirect framing (no inserted "
                "arm exists by construction); teacher-forced capture, layer 19",
                "fits": "held-out K=5 conversation-grouped folds (shared production fold map)",
                "arms": "context (prefix + user query) and prefix (everything before the query)",
            },
            "pair_classes_run": ["cross_character", "cross_model"],
            "note": "cross-framing pairs into/out of the indirect cells were deliberately not "
            "run: the indirect framing has no inserted arm, so on-policy cross-framing "
            "pairs are excluded per the plan's interpretive split.",
        },
        "indirect_ceilings": indirect,
        "banked_on_policy_ceilings": banked,
        "ceiling_summaries": {
            "indirect_context": ceil_summary(indirect, "context"),
            "indirect_prefix": ceil_summary(indirect, "prefix"),
            "banked_on_policy_context": ceil_summary(banked, "context"),
            "banked_on_policy_prefix": ceil_summary(banked, "prefix"),
        },
        "ladder_units": units,
        "ladder_summary": ladder_summary,
    }
    OUT_DIGEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIGEST.write_text(json.dumps(digest, indent=1, sort_keys=True))
    print(f"wrote {OUT_DIGEST}")
    print(json.dumps(digest["ceiling_summaries"], indent=1))
    for k, v in ladder_summary.items():
        print(
            k,
            {r: round(v["rung_r2_median"][r], 3) for r in RUNGS},
            "ceil",
            round(v["median_target_ceiling"], 3),
        )

    fig_ceilings(indirect, banked)
    fig_ladder_summary(units)
    fig_ladder_units(units)
    print("figures written under figures/" + OUT_FIG)


if __name__ == "__main__":
    main()
