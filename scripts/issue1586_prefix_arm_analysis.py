#!/usr/bin/env python
"""#1586 free-analysis: PREFIX-ARM method contrast (full FT vs LoRA).

9a-ter zero-GPU round. The pooled lattice (scripts/issue1586_pooled_lattice.py)
computed the prefix-arm seed-pooled paired FT-minus-LoRA mean-shift-norm
difference but the clean-result never interpreted it. This script RE-READS the
committed records only (no model calls, no new bootstrap recipe — every CI
below was produced by the existing machinery: paired question-cluster
bootstrap, n_boot=2000, seed 653, seed-stratified pooling):

- eval_results/issue_1586/geometry/pooled_lattice.json  (pooled per arm)
- eval_results/issue_1586/geometry/_beh_<key>_own_norm2000.json  (per-seed
  paired diff records — per-cell CIs for EVERY arm x layer already persisted,
  so no bootstrap-matrix re-reduction is needed)

Arm definitions (experiments.issue_1112.CAPTURE_ARMS): prefix = everything
before the user query; context = prefix + query; response = the model's own
answer tokens. Sign convention: point = ||mean(trained-base)||_FT -
||mean(trained-base)||_LoRA, so positive = full fine-tuning moves that arm's
activations farther from base than LoRA does.

Outputs:
- eval_results/issue_1586/geometry/prefix_arm_contrast.json
- figures/issue_1586/prefix_vs_context_arm_contrast.{png,pdf,meta.json}
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GEO = REPO_ROOT / "eval_results" / "issue_1586" / "geometry"
FIGDIR = REPO_ROOT / "figures" / "issue_1586"
OUT_JSON = GEO / "prefix_arm_contrast.json"

BEH_LAYER = {"syc": 14, "imp": 14, "cas": 14, "mk": 25}
BEH_NAME = {"syc": "sycophancy", "imp": "impolite", "cas": "casual style", "mk": "marker"}
REG_NAME = {"con": "contrastive", "po": "positive-only"}
ARMS = ("prefix", "context", "response")
SEEDS = ("s42", "s137")


def _sign(x: float) -> int:
    """Point-estimate sign as -1/0/+1."""
    return int(np.sign(x))


def _rec_view(rec: dict) -> dict:
    """Committed paired-diff record + derived sign flags (record copied verbatim)."""
    return {
        **rec,
        "point_sign": _sign(rec["point"]),
        "ci_excludes_zero": bool(rec["ci_low"] > 0 or rec["ci_high"] < 0),
    }


def load_inputs() -> tuple[dict, dict]:
    """(pooled norm records, per-seed diff records by behavior key)."""
    pooled = json.loads((GEO / "pooled_lattice.json").read_text())["norm"]
    per_seed = {
        beh: json.loads((GEO / f"_beh_{beh}_own_norm2000.json").read_text()) for beh in BEH_LAYER
    }
    return pooled, per_seed


def build_contrast(pooled: dict, per_seed: dict) -> dict:
    """Pooled + per-cell prefix/context/response reads + sign-consistency table.

    Asserts the pooled records' per_seed_points reconcile with the per-cell
    points from the norm2000 files (same clouds, two independent code paths).
    """
    out_pooled: dict[str, dict] = {}
    per_cell_rows: list[dict] = []
    per_group_signs: dict[str, dict] = {}
    for beh, layer in BEH_LAYER.items():
        for regime in ("con", "po"):
            gkey = f"{beh}/{regime}"
            grp: dict[str, dict] = {}
            for arm in ARMS:
                grp[arm] = _rec_view(pooled[f"own/{beh}/{regime}/{arm}/L{layer}"])
            grp["prefix_minus_context_point"] = grp["prefix"]["point"] - grp["context"]["point"]
            grp["prefix_context_sign_agree"] = (
                grp["prefix"]["point_sign"] == grp["context"]["point_sign"]
            )
            out_pooled[gkey] = grp
            cell_signs: dict[str, list[int]] = {arm: [] for arm in ARMS}
            for seed in SEEDS:
                entry = per_seed[beh]["diffs"][f"{beh}-pers-ft-{regime}-{seed}__ft_vs_lora"]
                row: dict = {
                    "behavior": BEH_NAME[beh],
                    "behavior_key": beh,
                    "regime": REG_NAME[regime],
                    "regime_key": regime,
                    "seed": seed,
                    "layer": layer,
                }
                for arm in ARMS:
                    rec = _rec_view(entry["reads"][f"{arm}/L{layer}"])
                    row[arm] = rec
                    cell_signs[arm].append(rec["point_sign"])
                # Reconcile against the pooled record's per-seed points
                # (pooled_norm_read vs norm_diff_pass — same clouds).
                si = SEEDS.index(seed)
                for arm in ARMS:
                    a = row[arm]["point"]
                    b = pooled[f"own/{beh}/{regime}/{arm}/L{layer}"]["per_seed_points"][si]
                    assert np.isclose(a, b, rtol=1e-4, atol=1e-3), (gkey, seed, arm, a, b)
                row["prefix_context_sign_agree"] = (
                    row["prefix"]["point_sign"] == row["context"]["point_sign"]
                )
                per_cell_rows.append(row)
            per_group_signs[gkey] = {
                "prefix_seed_signs": cell_signs["prefix"],
                "context_seed_signs": cell_signs["context"],
                "response_seed_signs": cell_signs["response"],
                "prefix_seeds_agree": len(set(cell_signs["prefix"])) == 1,
                "context_seeds_agree": len(set(cell_signs["context"])) == 1,
                "prefix_pooled_sign": grp["prefix"]["point_sign"],
                "context_pooled_sign": grp["context"]["point_sign"],
                "response_pooled_sign": grp["response"]["point_sign"],
                "prefix_pooled_ci_excludes_zero": grp["prefix"]["ci_excludes_zero"],
                "prefix_context_pooled_sign_agree": grp["prefix_context_sign_agree"],
                "prefix_response_pooled_sign_agree": (
                    grp["prefix"]["point_sign"] == grp["response"]["point_sign"]
                ),
            }
    groups = list(per_group_signs.values())
    summary = {
        "n_groups": len(groups),
        "n_per_seed_cells": len(per_cell_rows),
        "prefix_seeds_agree_groups": sum(g["prefix_seeds_agree"] for g in groups),
        "context_seeds_agree_groups": sum(g["context_seeds_agree"] for g in groups),
        "prefix_context_pooled_sign_agree_groups": sum(
            g["prefix_context_pooled_sign_agree"] for g in groups
        ),
        "prefix_response_pooled_sign_agree_groups": sum(
            g["prefix_response_pooled_sign_agree"] for g in groups
        ),
        "prefix_pooled_ci_excludes_zero_groups": sum(
            g["prefix_pooled_ci_excludes_zero"] for g in groups
        ),
    }
    return {
        "pooled": out_pooled,
        "per_cell": per_cell_rows,
        "sign_consistency": {"per_group": per_group_signs, "summary": summary},
    }


def make_figure(contrast: dict) -> None:
    """Forest plot: pooled prefix vs context FT-minus-LoRA Δnorm per behavior x
    regime, per-seed cells as labeled open markers; 95% cluster-bootstrap CIs."""
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.5, 9.0))
    arm_color = {"prefix": paper_palette_role("primary"), "context": paper_palette_role("baseline")}
    y = 0.0
    yticks, ylabels = [], []
    for beh in ("syc", "imp", "cas", "mk"):
        for regime in ("con", "po"):
            grp = contrast["pooled"][f"{beh}/{regime}"]
            cells = [
                r
                for r in contrast["per_cell"]
                if r["behavior_key"] == beh and r["regime_key"] == regime
            ]
            for arm in ("prefix", "context"):
                rec = grp[arm]
                # Clamp whisker lengths at 0 (float-epsilon / inverted-CI guard).
                xerr = [
                    [max(0.0, rec["point"] - rec["ci_low"])],
                    [max(0.0, rec["ci_high"] - rec["point"])],
                ]
                ax.errorbar(
                    rec["point"],
                    y,
                    xerr=xerr,
                    fmt="D",
                    color=arm_color[arm],
                    markersize=7,
                    capsize=3,
                    lw=1.8,
                    zorder=3,
                )
                for si, row in enumerate(cells):
                    r = row[arm]
                    yo = y + (0.28 if si == 0 else -0.28)
                    ax.errorbar(
                        r["point"],
                        yo,
                        xerr=[
                            [max(0.0, r["point"] - r["ci_low"])],
                            [max(0.0, r["ci_high"] - r["point"])],
                        ],
                        fmt="o",
                        color=arm_color[arm],
                        markerfacecolor="white",
                        markeredgewidth=1.1,
                        markersize=4.5,
                        capsize=1.5,
                        lw=0.8,
                        alpha=0.85,
                        zorder=2,
                    )
                    ax.annotate(
                        row["seed"],
                        (r["point"], yo),
                        textcoords="offset points",
                        xytext=(5, -2),
                        fontsize=6.5,
                        color=arm_color[arm],
                        alpha=0.9,
                    )
                yticks.append(y)
                ylabels.append(f"{BEH_NAME[beh]}, {REG_NAME[regime]} — {arm}")
                y += 1.0
            y += 0.9
        y += 0.5
    ax.axvline(0, color="0.4", lw=0.8, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Mean activation-shift norm difference, full fine-tune minus LoRA\n"
        "(own-text capture, registered layer; 95% paired cluster-bootstrap CI, 2000 draws)"
    )
    ax.set_title(
        "Prefix-arm vs context-arm method contrast (full fine-tune minus LoRA)\n"
        "(diamond = seed-pooled; labeled open circles = per-seed cells)",
        pad=16,
        fontsize=11,
    )
    handles = [
        plt.Line2D([], [], color=arm_color["prefix"], marker="D", ls="", label="prefix arm"),
        plt.Line2D([], [], color=arm_color["context"], marker="D", ls="", label="context arm"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    savefig_paper(fig, "prefix_vs_context_arm_contrast", dir=FIGDIR)
    plt.close(fig)


def main() -> int:
    """Build prefix_arm_contrast.json + the prefix-vs-context forest figure."""
    pooled, per_seed = load_inputs()
    contrast = build_contrast(pooled, per_seed)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    payload = {
        "metadata": {
            "script": "scripts/issue1586_prefix_arm_analysis.py",
            "git_commit": commit,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "numpy_version": np.__version__,
            "inputs": [
                "eval_results/issue_1586/geometry/pooled_lattice.json",
                *[
                    f"eval_results/issue_1586/geometry/_beh_{b}_own_norm2000.json"
                    for b in BEH_LAYER
                ],
            ],
            "conventions": {
                "statistic": "paired FT-minus-LoRA mean-shift-norm difference "
                "(||mean(trained-base)||_ft - ||mean(trained-base)||_lora)",
                "bootstrap": "paired question-cluster bootstrap, n_boot=2000, seed=653, "
                "seed-stratified pooling (records copied verbatim from the committed lattice)",
                "arms": {
                    "prefix": "everything before the user query",
                    "context": "prefix + user query",
                    "response": "model's own answer tokens",
                },
                "layers": BEH_LAYER,
                "sign": "positive = full fine-tuning moves the arm's activations farther "
                "from base than LoRA",
            },
        },
        **contrast,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1))
    print(f"[prefix-arm] wrote {OUT_JSON}")
    s = contrast["sign_consistency"]["summary"]
    print(f"[prefix-arm] sign-consistency summary: {json.dumps(s)}")
    make_figure(contrast)
    print(f"[prefix-arm] wrote {FIGDIR / 'prefix_vs_context_arm_contrast.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
