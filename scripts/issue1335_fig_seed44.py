"""Figures for issue #1335 follow-up round `seed44-base-rungs`.

Three-seed comparison of the base assistant-vs-fiction gap:
  1. seed_gap_trajectory   — gap + framing delta per generation seed (base 3 seeds,
                             instruct 2 seeds), 95% joint-draw CIs.
  2. seed_rungs_endpoint   — matched per-rung base values per seed (left) and
                             full-n per-persona fiction-endpoint values with CIs (right).
  3. collapse_vs_endpoint  — run-level under-floor rollout rate vs full-n endpoint
                             persona-mean R^2, one labeled point per seed.

Inputs (committed eval JSONs on branch issue-1335):
  eval_results/issue_1335/ladder_summary.json                    (seed 42 parent)
  eval_results/issue_1335/seed43-gap-rungs/seed_comparison.json  (seed 43 round)
  eval_results/issue_1335/seed44-base-rungs/seed_comparison.json (seed 44 round)
  eval_results/issue_1335/**/cells_r7_endpoint__base__<P>__ctx.json (full-n CIs)

Run from repo root: uv run python scripts/issue1335_fig_seed44.py
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "eval_results" / "issue_1335"
OUT_PREFIX = "issue_1335/seed44-base-rungs"

PERSONAS = ["Wren", "HELIOS", "Dana", "Vex"]
SEEDS = [42, 43, 44]

# Under-floor (<4-token) line rates on the base fiction-endpoint rollouts,
# re-derived from the raw rollout JSONLs (7,200 lines each):
#   seed 42: data/issue_1335/hf_dl/endpoint_base_gen.jsonl            -> 996/7200
#   seed 43: HF issue1335_ablation_ladder/seed43_gap_rungs/raw_completions/endpoint -> 272/7200
#   seed 44: HF issue1335_ablation_ladder/seed44_base_rungs/raw_completions/endpoint -> 1027/7200
UNDER_FLOOR_PCT = {42: 100 * 996 / 7200, 43: 100 * 272 / 7200, 44: 100 * 1027 / 7200}


def load_series():
    s42 = json.load(open(EV / "ladder_summary.json"))
    s43 = json.load(open(EV / "seed43-gap-rungs" / "seed_comparison.json"))
    s44 = json.load(open(EV / "seed44-base-rungs" / "seed_comparison.json"))

    def gv(d):  # {value, ci_lo, ci_hi} -> (v, lo, hi)
        return d["value"], d["ci_lo"], d["ci_hi"]

    series = {
        "base": {
            "gap": {
                42: gv(s42["per_model"]["base"]["gap"]["G"]),
                43: gv(s43["per_model"]["base"]["gap_G"]),
                44: gv(s44["per_model"]["base"]["gap_G"]),
            },
            "framing": {
                42: gv(s42["per_model"]["base"]["deltas"]["framing"]),
                43: gv(s43["per_model"]["base"]["framing"]),
                44: gv(s44["per_model"]["base"]["framing"]),
            },
        },
        "instruct": {
            "gap": {
                42: gv(s42["per_model"]["instruct"]["gap"]["G"]),
                43: gv(s43["per_model"]["instruct"]["gap_G"]),
            },
            "framing": {
                42: gv(s42["per_model"]["instruct"]["deltas"]["framing"]),
                43: gv(s43["per_model"]["instruct"]["framing"]),
            },
        },
    }
    matched = {
        42: s42["per_model"]["base"]["rung_values_matched_ctx"],
        43: s43["per_model"]["base"]["rung_values_matched_ctx"],
        44: s44["per_model"]["base"]["rung_values_matched_ctx"],
    }
    return series, matched


def load_endpoint_cells():
    dirs = {42: EV, 43: EV / "seed43-gap-rungs", 44: EV / "seed44-base-rungs"}
    out = {}
    for seed, d in dirs.items():
        out[seed] = {}
        for p in PERSONAS:
            j = json.load(open(d / f"cells_r7_endpoint__base__{p}__ctx.json"))
            gb = j["group_bootstrap_l19"]
            out[seed][p] = (gb["r2"], gb["ci_lo"], gb["ci_hi"])
    return out


def fig_gap_trajectory(series):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    panels = [
        ("gap", "R² gap (one-line Q&A − fiction endpoint)"),
        ("framing", "R² drop at fiction framing"),
    ]
    colors = {"base": paper_palette_role("primary"), "instruct": paper_palette_role("baseline")}
    for ax, (key, ylab) in zip(axes, panels):
        for model, off in (("base", -0.06), ("instruct", 0.06)):
            pts = series[model][key]
            xs = [i + off for i, s in enumerate(SEEDS) if s in pts]
            vs = [pts[s][0] for s in SEEDS if s in pts]
            los = [pts[s][0] - pts[s][1] for s in SEEDS if s in pts]
            his = [pts[s][2] - pts[s][0] for s in SEEDS if s in pts]
            label = "base model" if model == "base" else "instruct (seeds 42–43 only)"
            ax.errorbar(
                xs,
                vs,
                yerr=[los, his],
                fmt="o",
                color=colors[model],
                capsize=3,
                markersize=6,
                markeredgewidth=1.0,
                label=label,
            )
        ax.axhline(0.0, color="0.65", linewidth=0.8, zorder=0)
        ax.set_xticks(range(len(SEEDS)))
        ax.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax.set_xlabel("generation seed")
        ax.set_ylabel(ylab)
    axes[0].set_title("assistant-vs-fiction gap", loc="left")
    axes[1].set_title("fiction-framing delta", loc="left")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    savefig_paper(fig, f"{OUT_PREFIX}/seed_gap_trajectory")
    plt.close(fig)


def fig_rungs_endpoint(matched, cells):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax = axes[0]
    rungs = [
        ("r1_qa_oneline", "one-line Q&A"),
        ("r3_persona", "persona-described"),
        ("r4_fictionframe", "fiction-framed Q&A"),
        ("r7_endpoint_mean", "fiction endpoint (persona mean)"),
    ]
    from explore_persona_space.analysis.paper_plots import paper_palette_blog

    pal = paper_palette_blog(len(rungs))
    for (slug, label), c in zip(rungs, pal):
        vs = [matched[s][slug] for s in SEEDS]
        ax.plot(range(len(SEEDS)), vs, "-o", color=c, markersize=5, label=label)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_xlabel("generation seed")
    ax.set_ylabel("held-out R² (matched n, layer 19)")
    ax.set_title("matched per-rung values, base model", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    pal_p = paper_palette_blog(len(PERSONAS))
    for i, (p, c) in enumerate(zip(PERSONAS, pal_p)):
        xs = [j + (i - 1.5) * 0.05 for j in range(len(SEEDS))]
        vs = [cells[s][p][0] for s in SEEDS]
        los = [cells[s][p][0] - cells[s][p][1] for s in SEEDS]
        his = [cells[s][p][2] - cells[s][p][0] for s in SEEDS]
        ax.errorbar(
            xs,
            vs,
            yerr=[los, his],
            fmt="o-",
            color=c,
            capsize=2,
            markersize=4.5,
            linewidth=1.0,
            label=p,
        )
        ax.text(xs[-1] + 0.07, vs[-1], p, fontsize=8, color=c, va="center")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_xlabel("generation seed")
    ax.set_ylabel("held-out R² (full n, layer 19)")
    ax.set_title("fiction endpoint per persona, base model", loc="left")
    ax.set_xlim(-0.4, len(SEEDS) - 0.3)
    fig.tight_layout()
    savefig_paper(fig, f"{OUT_PREFIX}/seed_rungs_endpoint")
    plt.close(fig)


def fig_collapse(cells):
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for s in SEEDS:
        mean_r2 = sum(cells[s][p][0] for p in PERSONAS) / len(PERSONAS)
        x = UNDER_FLOOR_PCT[s]
        ax.scatter(
            [x],
            [mean_r2],
            s=55,
            color=paper_palette_role("primary"),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(x + 0.3, mean_r2, f"seed {s}", fontsize=9, va="center")
    ax.set_xlabel("under-4-token rollout lines (% of 7,200)")
    ax.set_ylabel("fiction endpoint R² (persona mean)")
    ax.set_title("rollout degeneracy rate vs endpoint map strength", loc="left")
    ax.set_xlim(0, 18)
    fig.tight_layout()
    savefig_paper(fig, f"{OUT_PREFIX}/collapse_vs_endpoint")
    plt.close(fig)


def main():
    set_paper_style("blog")
    series, matched = load_series()
    cells = load_endpoint_cells()
    fig_gap_trajectory(series)
    fig_rungs_endpoint(matched, cells)
    fig_collapse(cells)
    print("done")


if __name__ == "__main__":
    main()
