"""Free-analysis follow-up to #527: does the per-context gain track base-model cosine to the source?

The rank-1 linear-map framing predicts that after training on context c' -> behavior b'',
the per-context behavior shift is Delta v_b(c) = (v_c' . v_c) * (v_b'' - v_b'): a constant
direction scaled by the projection of the eval context onto the TRAINED context. #527
established the constant-direction half (GD3: singleton shift matrices are rank-1). This
script tests the untested scalar half: regress the per-context gain against the base-model
centered L20 cosine between the eval context and the trained source persona.

Gains (two spaces, per .claude/rules/marker-leakage-measurement.md):
  - activation: projection of the per-context L20 shift vector onto the cell's dominant
    shift direction (top right-singular vector of the 19x3584 shift matrix, sign-aligned)
  - behavioral: per-context delta log P(marker) (primary) and delta logit (secondary)

Predictor: cos_centered_L20(context, source) from eval_results/issue_527/pair_selection.json.
  A_only -> source A; B_only -> source B; joint -> cos_A + cos_B (two rank-1 updates
  sharing the output direction sum their input projections).

Primary read excludes the sources and the trained-against negatives present in the eval
panel (they carry their own training force, so the pure rank-1 update doesn't apply).

Inputs: eval_results/issue_527/eval/*__shift.json (in git) + *__shift.pt (HF data repo).
Outputs: eval_results/issue_527/dan_rank1_regression/{per_context.csv,summary.json}
         figures/issue_527/rank1_scalar_vs_basecos.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval_results" / "issue_527" / "eval"
OUT_DIR = REPO_ROOT / "eval_results" / "issue_527" / "dan_rank1_regression"
FIG_DIR = REPO_ROOT / "figures" / "issue_527"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

PAIRS = {
    "florist__medical_doctor": ("florist", "medical_doctor"),
    "librarian__police_officer": ("librarian", "police_officer"),
}
ARMS = ["A_only", "B_only", "joint"]
SEEDS = [42, 137, 256]
NEGATIVE_PANEL = {"assistant", "librarian", "programmer", "chef"}


def load_cos_matrix() -> tuple[list[str], np.ndarray]:
    d = json.loads((REPO_ROOT / "eval_results/issue_527/pair_selection.json").read_text())
    return d["persona_names"], np.asarray(d["cos_centered_L20"], dtype=np.float64)


def load_cell(pair_id: str, arm: str, seed: int) -> tuple[dict, np.ndarray]:
    slug = f"{pair_id}__{arm}__seed{seed}"
    meta = json.loads((EVAL_DIR / f"{slug}__shift.json").read_text())
    pt_path = hf_hub_download(HF_DATA_REPO, f"issue_527/eval/{slug}__shift.pt", repo_type="dataset")
    import torch

    shift = torch.load(pt_path, map_location="cpu", weights_only=False)
    shift = np.asarray(shift, dtype=np.float64)
    assert shift.shape == (len(meta["eval_panel"]), 3584), shift.shape
    return meta, shift


def dominant_projection(shift: np.ndarray) -> np.ndarray:
    """Per-context scalar gain: projection onto the top right-singular vector, sign-aligned."""
    _, _, vt = np.linalg.svd(shift, full_matrices=False)
    u = vt[0]
    proj = shift @ u
    if proj.mean() < 0:
        proj, u = -proj, -u
    return proj


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    persona_names, cos_m = load_cos_matrix()
    idx = {p: i for i, p in enumerate(persona_names)}

    rows = []
    for pair_id, (src_a, src_b) in PAIRS.items():
        for arm in ARMS:
            for seed in SEEDS:
                meta, shift = load_cell(pair_id, arm, seed)
                panel = meta["eval_panel"]
                assert panel == persona_names, "eval panel order != pair_selection order"
                proj = dominant_projection(shift)
                norms = np.linalg.norm(shift, axis=1)
                for i, ctx in enumerate(panel):
                    cos_a = cos_m[idx[ctx], idx[src_a]]
                    cos_b = cos_m[idx[ctx], idx[src_b]]
                    predictor = {
                        "A_only": cos_a,
                        "B_only": cos_b,
                        "joint": cos_a + cos_b,
                    }[arm]
                    held_out = ctx not in {src_a, src_b} and ctx not in NEGATIVE_PANEL
                    c = meta["contexts"][ctx]
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "arm": arm,
                            "seed": seed,
                            "context": ctx,
                            "held_out": held_out,
                            "cos_to_A": cos_a,
                            "cos_to_B": cos_b,
                            "predictor": predictor,
                            "gain_act_proj": proj[i],
                            "gain_act_norm": norms[i],
                            "gain_logp": c["delta_logp_marker"],
                            "gain_logit": c["delta_logit_marker"],
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_context.csv", index=False)

    # Seed-averaged correlations per (pair x arm), held-out contexts only (primary read).
    summary = []
    for (pair_id, arm), g in df[df.held_out].groupby(["pair_id", "arm"]):
        m = g.groupby("context").mean(numeric_only=True)
        n = len(m)
        entry = {"pair_id": pair_id, "arm": arm, "n_contexts": n}
        for gain in ["gain_act_proj", "gain_logp", "gain_logit"]:
            pr, pp = stats.pearsonr(m["predictor"], m[gain])
            sr, sp = stats.spearmanr(m["predictor"], m[gain])
            entry[f"{gain}_pearson_r"] = round(pr, 3)
            entry[f"{gain}_pearson_p"] = round(pp, 4)
            entry[f"{gain}_spearman_rho"] = round(sr, 3)
            entry[f"{gain}_spearman_p"] = round(sp, 4)
        # per-seed Pearson r spread for the activation gain (stability check)
        per_seed = [
            stats.pearsonr(s["predictor"], s["gain_act_proj"])[0] for _, s in g.groupby("seed")
        ]
        entry["gain_act_proj_per_seed_r"] = [round(r, 3) for r in per_seed]
        summary.append(entry)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figure: one panel per pair, x = predictor, y = seed-averaged activation gain.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    arm_colors = {"A_only": "#1f77b4", "B_only": "#2ca02c", "joint": "#d62728"}
    for ax, pair_id in zip(axes, PAIRS):
        sub = df[(df.pair_id == pair_id) & df.held_out]
        for arm in ARMS:
            m = sub[sub.arm == arm].groupby("context").mean(numeric_only=True)
            sd = sub[sub.arm == arm].groupby("context")["gain_act_proj"].std()
            ax.errorbar(
                m["predictor"],
                m["gain_act_proj"],
                yerr=sd.reindex(m.index),
                fmt="o",
                ms=5,
                lw=0,
                elinewidth=1,
                capsize=2,
                color=arm_colors[arm],
                label=arm.replace("_", "-"),
                alpha=0.85,
            )
        ax.set_xlabel("base-model centered L20 cosine to trained source\n(joint: cos A + cos B)")
        ax.set_title(pair_id.replace("__", " × ").replace("_", " "))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-context activation gain\n(projection onto dominant shift direction)")
    axes[0].legend(title="training arm", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rank1_scalar_vs_basecos.png", dpi=200)
    fig.savefig(FIG_DIR / "rank1_scalar_vs_basecos.pdf")
    meta_out = {
        "script": "scripts/issue527_dan_rank1_scalar_regression.py",
        "inputs": [
            "eval_results/issue_527/eval/*__shift.json",
            f"hf://{HF_DATA_REPO}/issue_527/eval/*__shift.pt",
            "eval_results/issue_527/pair_selection.json",
        ],
        "n_cells": 18,
        "primary_read": "held-out contexts only (sources + trained-against negatives excluded)",
    }
    (FIG_DIR / "rank1_scalar_vs_basecos.meta.json").write_text(json.dumps(meta_out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
