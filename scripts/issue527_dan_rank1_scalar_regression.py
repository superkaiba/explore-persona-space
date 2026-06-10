"""Rank-1 scalar regression: does the per-context gain track base cosine to the trained source?

The rank-1 linear-map framing predicts that after training on context c' -> behavior b'',
the per-context behavior shift is Delta v_b(c) = (v_c' . v_c) * (v_b'' - v_b'): a constant
direction scaled by the projection of the eval context onto the TRAINED context. #527
established the constant-direction half (GD3: singleton shift matrices are rank-1). This
script tests the scalar half: regress the per-context gain against the base-model centered
L20 cosine between the eval context and the trained source persona.

Gains (two spaces, per .claude/rules/marker-leakage-measurement.md):
  - activation: projection of the per-context L20 shift vector onto the cell's dominant
    shift direction (top right-singular vector of the 19x3584 shift matrix, sign-aligned)
  - behavioral: per-context delta log P(marker) (primary) and delta logit (secondary)

Predictor: cos_centered_L20(context, source) from eval_results/issue_527/pair_selection.json
(#538 inherits #527's pair selection verbatim).
  A_only -> source A; B_only -> source B; joint -> cos_A + cos_B (two rank-1 updates
  sharing the output direction sum their input projections).

Primary read excludes the sources and the trained-against negatives present in the eval
panel (they carry their own training force, so the pure rank-1 update doesn't apply).

Modes:
  --issue 527 (default) | 538   per-issue regression over that issue's shift extractions
                                (#538 is the [14,20]-nat-band re-run of #527's [5,12])
  --compare                     two-point dose contrast: fit gain = a + b*cos per
                                (pair x arm) at each dose, compare slope/intercept across
                                the two bands. Requires both per-issue runs first.

Outputs (per-issue): eval_results/issue_<N>/dan_rank1_regression/{per_context.csv,summary.json}
                     figures/issue_<N>/rank1_scalar_vs_basecos.{png,pdf}
Outputs (compare):   eval_results/issue_538/dan_rank1_regression/dose_contrast.json
                     figures/issue_538/rank1_dose_contrast.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

PAIRS = {
    "florist__medical_doctor": ("florist", "medical_doctor"),
    "librarian__police_officer": ("librarian", "police_officer"),
}
ARMS = ["A_only", "B_only", "joint"]
SEEDS = [42, 137, 256]
NEGATIVE_PANEL = {"assistant", "librarian", "programmer", "chef"}
BAND_LABEL = {527: "[5,12] nat band", 538: "[14,20] nat band"}


def eval_dir(issue: int) -> Path:
    return REPO_ROOT / "eval_results" / f"issue_{issue}" / "eval"


def out_dir(issue: int) -> Path:
    return REPO_ROOT / "eval_results" / f"issue_{issue}" / "dan_rank1_regression"


def fig_dir(issue: int) -> Path:
    return REPO_ROOT / "figures" / f"issue_{issue}"


def load_cos_matrix() -> tuple[list[str], np.ndarray]:
    # #538 inherits #527's pair selection (READ prefix stays issue_527).
    d = json.loads((REPO_ROOT / "eval_results/issue_527/pair_selection.json").read_text())
    return d["persona_names"], np.asarray(d["cos_centered_L20"], dtype=np.float64)


def load_cell(issue: int, pair_id: str, arm: str, seed: int) -> tuple[dict, np.ndarray]:
    slug = f"{pair_id}__{arm}__seed{seed}"
    meta = json.loads((eval_dir(issue) / f"{slug}__shift.json").read_text())
    local_pt = eval_dir(issue) / f"{slug}__shift.pt"
    if local_pt.exists():
        pt_path = str(local_pt)
    else:
        from huggingface_hub import hf_hub_download

        pt_path = hf_hub_download(
            HF_DATA_REPO, f"issue_{issue}/eval/{slug}__shift.pt", repo_type="dataset"
        )
    import torch

    shift = np.asarray(
        torch.load(pt_path, map_location="cpu", weights_only=False), dtype=np.float64
    )
    assert shift.shape == (len(meta["eval_panel"]), 3584), shift.shape
    return meta, shift


def dominant_projection(shift: np.ndarray) -> np.ndarray:
    """Per-context scalar gain: projection onto the top right-singular vector, sign-aligned."""
    _, _, vt = np.linalg.svd(shift, full_matrices=False)
    u = vt[0]
    proj = shift @ u
    if proj.mean() < 0:
        proj = -proj
    return proj


def run_issue(issue: int) -> None:
    out, figs = out_dir(issue), fig_dir(issue)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    persona_names, cos_m = load_cos_matrix()
    idx = {p: i for i, p in enumerate(persona_names)}

    rows = []
    for pair_id, (src_a, src_b) in PAIRS.items():
        for arm in ARMS:
            for seed in SEEDS:
                meta, shift = load_cell(issue, pair_id, arm, seed)
                panel = meta["eval_panel"]
                assert panel == persona_names, "eval panel order != pair_selection order"
                proj = dominant_projection(shift)
                norms = np.linalg.norm(shift, axis=1)
                for i, ctx in enumerate(panel):
                    cos_a = cos_m[idx[ctx], idx[src_a]]
                    cos_b = cos_m[idx[ctx], idx[src_b]]
                    predictor = {"A_only": cos_a, "B_only": cos_b, "joint": cos_a + cos_b}[arm]
                    held_out = ctx not in {src_a, src_b} and ctx not in NEGATIVE_PANEL
                    c = meta["contexts"][ctx]
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "arm": arm,
                            "seed": seed,
                            "context": ctx,
                            "held_out": held_out,
                            "is_source": ctx in {src_a, src_b},
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
    df.to_csv(out / "per_context.csv", index=False)

    # Seed-averaged correlations per (pair x arm), held-out contexts only (primary read).
    summary = []
    for (pair_id, arm), g in df[df.held_out].groupby(["pair_id", "arm"]):
        m = g.groupby("context").mean(numeric_only=True)
        entry = {"pair_id": pair_id, "arm": arm, "n_contexts": len(m)}
        for gain in ["gain_act_proj", "gain_logp", "gain_logit"]:
            pr, pp = stats.pearsonr(m["predictor"], m[gain])
            sr, sp = stats.spearmanr(m["predictor"], m[gain])
            entry[f"{gain}_pearson_r"] = round(pr, 3)
            entry[f"{gain}_pearson_p"] = round(pp, 4)
            entry[f"{gain}_spearman_rho"] = round(sr, 3)
            entry[f"{gain}_spearman_p"] = round(sp, 4)
        per_seed = [
            stats.pearsonr(s["predictor"], s["gain_act_proj"])[0] for _, s in g.groupby("seed")
        ]
        entry["gain_act_proj_per_seed_r"] = [round(r, 3) for r in per_seed]
        summary.append(entry)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Figure: one panel per pair, x = predictor, y = seed-averaged activation gain.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
    arm_colors = {"A_only": "#1f77b4", "B_only": "#2ca02c", "joint": "#d62728"}
    for ax, pair_id in zip(axes, PAIRS, strict=False):
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
        ax.set_title(f"{pair_id.replace('__', ' x ').replace('_', ' ')} - {BAND_LABEL[issue]}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-context activation gain\n(projection onto dominant shift direction)")
    axes[0].legend(title="training arm", frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "rank1_scalar_vs_basecos.png", dpi=200)
    fig.savefig(figs / "rank1_scalar_vs_basecos.pdf")
    (figs / "rank1_scalar_vs_basecos.meta.json").write_text(
        json.dumps(
            {
                "script": "scripts/issue527_dan_rank1_scalar_regression.py",
                "issue": issue,
                "inputs": [
                    f"eval_results/issue_{issue}/eval/*__shift.json",
                    f"hf://{HF_DATA_REPO}/issue_{issue}/eval/*__shift.pt",
                    "eval_results/issue_527/pair_selection.json",
                ],
                "n_cells": 18,
                "primary_read": "held-out contexts only"
                " (sources + trained-against negatives excluded)",
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


def _fit(df: pd.DataFrame, gain: str) -> dict:
    """Seed-averaged OLS gain = a + b*predictor on held-out contexts, plus per-seed slopes."""
    m = df.groupby("context").mean(numeric_only=True)
    b, a = np.polyfit(m["predictor"], m[gain], 1)
    per_seed_b = [np.polyfit(s["predictor"], s[gain], 1)[0] for _, s in df.groupby("seed")]
    return {
        "intercept": round(a, 3),
        "slope": round(b, 3),
        "slope_over_intercept": round(b / a, 3) if a != 0 else None,
        "per_seed_slope": [round(x, 3) for x in per_seed_b],
    }


def run_compare() -> None:
    frames = {}
    for issue in (527, 538):
        csv = out_dir(issue) / "per_context.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} missing — run `--issue {issue}` first")
        frames[issue] = pd.read_csv(csv)

    contrast = []
    for pair_id in PAIRS:
        for arm in ARMS:
            entry = {"pair_id": pair_id, "arm": arm}
            for issue in (527, 538):
                df = frames[issue]
                cell = df[(df.pair_id == pair_id) & (df.arm == arm)]
                held = cell[cell.held_out]
                # realized dose = seed-mean source-context delta log P (joint: mean of A,B)
                dose = cell[cell.is_source]["gain_logp"].mean()
                entry[f"issue{issue}"] = {
                    "band": BAND_LABEL[issue],
                    "realized_source_dose_nats": round(dose, 2),
                    "fit_logp": _fit(held, "gain_logp"),
                    "fit_act_proj": _fit(held, "gain_act_proj"),
                }
            contrast.append(entry)

    out = out_dir(538)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dose_contrast.json").write_text(json.dumps(contrast, indent=2))

    # Figure: intercept (left) and slope (right) of the log-prob fit vs realized dose,
    # one line per (pair x arm) connecting the two dose points.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    arm_colors = {"A_only": "#1f77b4", "B_only": "#2ca02c", "joint": "#d62728"}
    pair_markers = {"florist__medical_doctor": "o", "librarian__police_officer": "s"}
    for e in contrast:
        doses = [e[f"issue{i}"]["realized_source_dose_nats"] for i in (527, 538)]
        for ax, key in zip(axes, ["intercept", "slope"], strict=True):
            vals = [e[f"issue{i}"]["fit_logp"][key] for i in (527, 538)]
            ax.plot(
                doses,
                vals,
                marker=pair_markers[e["pair_id"]],
                color=arm_colors[e["arm"]],
                alpha=0.85,
                label=f"{e['pair_id'].split('__')[0]}-pair {e['arm'].replace('_', '-')}",
            )
    axes[0].set_ylabel("intercept a (nats)\ncontext-independent component")
    axes[1].set_ylabel("slope b (nats per unit cosine)\ncontext-projection component")
    for ax in axes:
        ax.set_xlabel("realized source dose (Δ log P at source, nats)")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("gain(c) = a + b·cos(v_src, v_c): dose contrast, [5,12] vs [14,20] band")
    fig.tight_layout()
    figs = fig_dir(538)
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "rank1_dose_contrast.png", dpi=200)
    fig.savefig(figs / "rank1_dose_contrast.pdf")
    (figs / "rank1_dose_contrast.meta.json").write_text(
        json.dumps(
            {
                "script": "scripts/issue527_dan_rank1_scalar_regression.py --compare",
                "inputs": [
                    "eval_results/issue_527/dan_rank1_regression/per_context.csv",
                    "eval_results/issue_538/dan_rank1_regression/per_context.csv",
                ],
            },
            indent=2,
        )
    )
    print(json.dumps(contrast, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--issue", type=int, choices=[527, 538], default=527)
    ap.add_argument("--compare", action="store_true", help="two-point dose contrast (527 vs 538)")
    args = ap.parse_args()
    if args.compare:
        run_compare()
    else:
        run_issue(args.issue)


if __name__ == "__main__":
    main()
