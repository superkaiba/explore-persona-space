"""MATS 2026 poster figure 11: model-size scaling + chain-of-thought.

Two separate wide-short figures (the two halves have different DVs and
corpora, so they do not share a panel):

``plot11_scaling_cot`` — the context->answer ridge map across the
Qwen2.5-Instruct scale ladder (0.5B/1.5B/3B/7B/14B/32B, issue #1491):
held-out test R^2 (variance-weighted, LMSYS+WildChat, matched train
n=25,000, test n=1,000) with 95% bootstrap CIs, the two-draw reliability
ceiling, and the ceiling-normalized read.

``plot11b_cot_reasoning`` — three thinking models (OpenThinker2-7B #928,
DeepSeek-R1-Distill-Qwen-7B #1005, DeepSeek-R1-Distill-Llama-8B #1426):
held-out skill on the shared answer-REMAINDER target, per-question regime
at the frozen primary layer, for context-only vs context + length-matched
answer-prefix vs context + length-matched truncated CoT vs context + full
CoT (95% paired-bootstrap CIs, n_boot=2000).

Numbers read ONLY from committed eval_results JSONs (never hand-typed):
- eval_results/issue_1491/scale_ladder/fits_<slug>.json
- eval_results/issue_1491/scale_ladder/caphit_restriction_<slug>.json
- eval_results/issue_928/matched-length-answer-span-control/mlc_bootstrap_deltaskill.json
- eval_results/issue_1005/mlc_bootstrap_deltaskill.json
- eval_results/issue_1426/mlc_bootstrap_deltaskill.json

Writes docs/posters/mats_2026/figures/plot11_scaling_cot.{png,pdf,meta.json},
plot11b_cot_reasoning.{png,pdf,meta.json}, and plot11_scaling_cot_data.json
(every plotted number + its source path).
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
LADDER = REPO / "eval_results" / "issue_1491" / "scale_ladder"
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"

# (fits/digest slug, params in billions, tick label) — Qwen2.5-Instruct ladder, #1491.
RUNGS = [
    ("scale05", 0.5, "0.5B"),
    ("scale15", 1.5, "1.5B"),
    ("scale3", 3.0, "3B"),
    ("scale7_refit", 7.0, "7B"),
    ("scale14", 14.0, "14B"),
    ("scale32", 32.0, "32B"),
]
CEILING_EXPECTED_N = 1000  # #2130 read-side defense: refuse a short/partial ceiling.

# (issue, mlc JSON path relative to repo, tick label, model id) — thinking models.
COT_MODELS = [
    (
        928,
        "eval_results/issue_928/matched-length-answer-span-control/mlc_bootstrap_deltaskill.json",
        "OpenThinker2\n7B",
        "open-thoughts/OpenThinker2-7B",
    ),
    (
        1005,
        "eval_results/issue_1005/mlc_bootstrap_deltaskill.json",
        "R1-Distill\nQwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ),
    (
        1426,
        "eval_results/issue_1426/mlc_bootstrap_deltaskill.json",
        "R1-Distill\nLlama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ),
]
# (absolute_at_frozen key, legend label)
COT_ARMS = [
    ("mlc_ctx", "context only"),
    ("mlc_ctx_apfx", "+ answer prefix (matched length)"),
    ("mlc_ctx_cotK", "+ truncated CoT (matched length)"),
    ("mlc_ctx_cotfull", "+ full CoT"),
]


def _load_ladder():
    rows = []
    for slug, params_b, label in RUNGS:
        fits = json.loads((LADDER / f"fits_{slug}.json").read_text())
        digest = json.loads((LADDER / f"caphit_restriction_{slug}.json").read_text())
        cd = fits["ceiling_two_draw"]
        if cd.get("n_pairs") != CEILING_EXPECTED_N:
            raise RuntimeError(
                f"{slug}: ceiling_two_draw.n_pairs={cd.get('n_pairs')} != "
                f"{CEILING_EXPECTED_N} — short/partial ceiling in committed fits JSON"
            )
        rf = digest["restriction"]["ridge_full"]
        lo, hi = rf["bootstrap"]["ci95"]
        r2 = rf["test_r2_refit"]
        rows.append(
            {
                "slug": slug,
                "model": fits["model"],
                "params_b": params_b,
                "label": label,
                "ridge_test_r2": r2,
                "ridge_ci95": [lo, hi],
                "ceiling_two_draw": cd["ceiling_var_weighted_r"],
                "shuffled_pairing_null": fits["floors"]["shuffled_pairing"]["test_r2"],
                "n_train": fits["n_realized"]["train_25k"],
                "n_test": fits["n_realized"]["test_1000"],
                "source_fits": f"eval_results/issue_1491/scale_ladder/fits_{slug}.json",
                "source_ci": f"eval_results/issue_1491/scale_ladder/caphit_restriction_{slug}.json",
            }
        )
    return rows


def _load_cot():
    rows = []
    for issue, rel, label, model_id in COT_MODELS:
        d = json.loads((REPO / rel).read_text())
        indiv = d["by_regime"]["indiv"]
        arms = {}
        for key, _ in COT_ARMS:
            a = indiv["absolute_at_frozen"][key]
            arms[key] = {"observed": a["observed"], "ci95": a["ci95"], "n_draws": a["n_draws"]}
        read1 = indiv["statistics"]["read1_primary_ctx_cotK_minus_ctx_apfx"][
            "primary_frozen_ctx_baseline_best"
        ]
        rows.append(
            {
                "issue": issue,
                "model": model_id,
                "label": label,
                "arms": arms,
                "cot_minus_prefix": {"observed": read1["observed"], "ci95": read1["ci95"]},
                "source": rel,
            }
        )
    return rows


def _yerr(center, ci):
    return [[center - ci[0]], [ci[1] - center]]


def fig_scaling(rows):
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    x = np.array([r["params_b"] for r in rows])
    ridge = np.array([r["ridge_test_r2"] for r in rows])
    ceil = np.array([r["ceiling_two_draw"] for r in rows])
    lo = np.array([r["ridge_ci95"][0] for r in rows])
    hi = np.array([r["ridge_ci95"][1] for r in rows])
    yerr = np.vstack([ridge - lo, hi - ridge])

    ax.plot(
        x,
        ceil,
        color=paper_color("reference"),
        linestyle="--",
        marker="s",
        markersize=3.5,
        linewidth=1.2,
        label="two-draw reliability ceiling",
    )
    ax.errorbar(
        x,
        ridge,
        yerr=yerr,
        color=paper_color("instruct"),
        marker="o",
        capsize=2.5,
        linewidth=1.6,
        label="ridge map, held-out test $R^2$",
        zorder=4,
    )
    ax.errorbar(
        x,
        ridge / ceil,
        yerr=yerr / ceil,
        color=paper_color("identity_bias"),
        marker="^",
        capsize=2.5,
        linewidth=1.4,
        label="$R^2$ ÷ ceiling",
        zorder=3,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [r["label"] for r in rows])
    ax.minorticks_off()
    ax.set_xlabel("model size (Qwen2.5-Instruct)")
    ax.set_ylabel("held-out test $R^2$")
    ax.set_ylim(0.45, 1.0)
    ax.legend(loc="lower right", fontsize=8)

    savefig_paper(fig, "plot11_scaling_cot", dir=OUT_DIR)
    plt.close(fig)


def fig_cot(rows):
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    n_arm = len(COT_ARMS)
    width = 0.19
    xs = np.arange(len(rows))
    colors = {
        "mlc_ctx": paper_color("instruct"),
        "mlc_ctx_apfx": paper_color("oracle_answer"),
        "mlc_ctx_cotK": paper_color("neural_map"),
        "mlc_ctx_cotfull": paper_color("neural_map"),
    }
    alphas = {"mlc_ctx_cotfull": 0.45}
    for j, (key, label) in enumerate(COT_ARMS):
        offs = (j - (n_arm - 1) / 2) * width
        vals = np.array([r["arms"][key]["observed"] for r in rows])
        lo = np.array([r["arms"][key]["ci95"][0] for r in rows])
        hi = np.array([r["arms"][key]["ci95"][1] for r in rows])
        ax.bar(
            xs + offs,
            vals,
            width=width * 0.92,
            color=colors[key],
            alpha=alphas.get(key, 1.0),
            yerr=np.vstack([vals - lo, hi - vals]),
            capsize=2,
            error_kw={"linewidth": 0.9},
            label=label,
        )
    ax.set_xticks(xs, [r["label"] for r in rows])
    ax.set_ylabel("held-out skill\n(answer-remainder target)")
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=8,
        frameon=False,
        borderaxespad=0.1,
    )

    savefig_paper(fig, "plot11b_cot_reasoning", dir=OUT_DIR)
    plt.close(fig)


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    ladder = _load_ladder()
    cot = _load_cot()

    fig_scaling(ladder)
    fig_cot(cot)

    data = {
        "figure_plot11_scaling_cot": {
            "dv": (
                "held-out test R^2 (variance-weighted, #779 metric) of the ridge "
                "context->answer map; LMSYS+WildChat contexts, each rung's own "
                "on-policy responses; matched train n=25,000, test n=1,000; error "
                "bars = 95% bootstrap CI (n_boot=1,000); ceiling = two-draw "
                "reliability (n_pairs=1,000); issue #1491"
            ),
            "rungs": ladder,
        },
        "figure_plot11b_cot_reasoning": {
            "dv": (
                "held-out skill-over-mean R^2 on the shared answer-REMAINDER "
                "target, per-question regime, frozen primary layer (context-only "
                "baseline's full-data best LOCO layer), LOCO folds over 50 shared "
                "contexts x 48 probes; 95% paired-bootstrap CIs (n_boot=2,000, "
                "seed 42); issues #928 / #1005 / #1426"
            ),
            "models": cot,
        },
    }
    out = OUT_DIR / "plot11_scaling_cot_data.json"
    out.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT_DIR}/plot11_scaling_cot.{{png,pdf,meta.json}}")
    print(f"wrote {OUT_DIR}/plot11b_cot_reasoning.{{png,pdf,meta.json}}")
    print(f"wrote {out}")
    for r in ladder:
        ratio = r["ridge_test_r2"] / r["ceiling_two_draw"]
        print(
            f"  {r['label']:>5} ridge={r['ridge_test_r2']:.4f} "
            f"ceil={r['ceiling_two_draw']:.4f} ratio={ratio:.4f}"
        )
    for r in cot:
        a = r["arms"]
        print(
            f"  #{r['issue']} ctx={a['mlc_ctx']['observed']:.3f} "
            f"apfx={a['mlc_ctx_apfx']['observed']:.3f} "
            f"cotK={a['mlc_ctx_cotK']['observed']:.3f} "
            f"cotfull={a['mlc_ctx_cotfull']['observed']:.3f} "
            f"cot-prefix={r['cot_minus_prefix']['observed']:+.4f}"
        )


if __name__ == "__main__":
    main()
