# ruff: noqa: RUF001
"""Layer-sweep geometry-vs-behavior figure for #1415 (9a-ter fold).

Three panels: (A) per-layer geometric answer-shift alignment (shared solid /
disjoint dashed, both arms); (B) per-layer mean graded judge shift vs baseline
(+/- SE over 28 pairs, both arms; L20 from the primary judged file); (C)
per-pair judge shifts per layer (both arms, symlog y). Writes a per-arm
per-layer summary JSON alongside.

Inputs: eval_results/issue_1415/{behavioral_judge_scores_layer_sweep,
behavioral_judge_scores,disjoint_baseline_recount}.json
Output figure goes to the MAIN checkout's figures/issue_1415/ (committed to main).
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)

WT_ROOT = Path(__file__).resolve().parent.parent
MAIN_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=WT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
).parent
EVAL = WT_ROOT / "eval_results" / "issue_1415"
FIGDIR = MAIN_ROOT / "figures" / "issue_1415"
LAYERS = [7, 10, 14, 17, 20, 21, 24]
SWEEP_LAYERS = [7, 10, 14, 17, 21, 24]
ARMS = ["prefix", "context"]

set_paper_style("blog")
C = paper_palette_blog(6)
ARM_COLOR = {"prefix": C[0], "context": C[1]}


def _per_pair_means(items: dict, arm_key: str, alpha: float | None = None) -> dict[str, float]:
    """Mean graded score per pair for one arm (None-scored items dropped)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for v in items.values():
        if v["arm"] != arm_key or v["graded_score"] is None:
            continue
        if alpha is not None and v.get("alpha") != alpha:
            continue
        acc[v["pair_id"]].append(v["graded_score"])
    return {p: float(np.mean(s)) for p, s in acc.items()}


def main() -> None:
    sweep = json.loads((EVAL / "behavioral_judge_scores_layer_sweep.json").read_text())["per_item"]
    primary = json.loads((EVAL / "behavioral_judge_scores.json").read_text())["per_item"]
    rc = json.loads((EVAL / "disjoint_baseline_recount.json").read_text())

    base = _per_pair_means(primary, "baseline")
    pairs = sorted(base)
    assert len(pairs) == 28, len(pairs)

    # per (arm, layer) pair-shift dict; L20 comes from the primary file at alpha=4
    shifts: dict[tuple[str, int], dict[str, float]] = {}
    for arm in ARMS:
        for lyr in SWEEP_LAYERS:
            pm = _per_pair_means(sweep, f"steered_L{lyr}_{arm}")
            shifts[(arm, lyr)] = {p: pm[p] - base[p] for p in pairs}
        pm20 = _per_pair_means(primary, f"steered_primary_{arm}", alpha=4.0)
        shifts[(arm, 20)] = {p: pm20[p] - base[p] for p in pairs}

    summary: dict[str, dict] = {}
    for (arm, lyr), sh in shifts.items():
        vals = np.array([sh[p] for p in pairs])
        _, pval = wilcoxon(vals, alternative="greater")
        summary[f"{arm}_L{lyr}"] = {
            "mean_shift": float(vals.mean()),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "wilcoxon_p_greater": float(pval),
            "matched_mean": float(np.mean([sh[p] for p in pairs if p.startswith("m")])),
            "cross_mean": float(np.mean([sh[p] for p in pairs if p.startswith("cross")])),
            "per_pair_shift": {p: float(sh[p]) for p in pairs},
            "source": "behavioral_judge_scores.json (primary, alpha=4)"
            if lyr == 20
            else "behavioral_judge_scores_layer_sweep.json",
        }
    out_json = EVAL / "layer_sweep_behavioral_summary.json"
    out_json.write_text(json.dumps(summary, indent=1))
    print(f"wrote {out_json}")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # --- Panel A: geometric per-layer alignment (matched steer/read layer) ---
    ax = axes[0]
    for arm in ARMS:
        pl = rc["h1"][arm]["per_layer_mean"]
        xs = [int(k) for k in sorted(pl["shared"], key=int)]
        ax.plot(
            xs,
            [pl["shared"][str(x)] for x in xs],
            "-o",
            color=ARM_COLOR[arm],
            label=f"{arm}-based V_c (shared)",
        )
        ax.plot(
            xs,
            [pl["disj"][str(x)] for x in xs],
            "--s",
            color=ARM_COLOR[arm],
            alpha=0.65,
            label=f"{arm}-based V_c (disjoint)",
        )
    ax.axhline(0.04, ls=":", color="gray", label="random-direction null p97.5")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Steer/read layer (matched)")
    ax.set_ylabel("Answer-shift alignment (cosine)")
    ax.set_title("Geometry: alignment peaks at layers 14–17")
    ax.legend(fontsize=8)

    # --- Panel B: behavioral mean shift per layer ---
    ax = axes[1]
    for arm in ARMS:
        means = [summary[f"{arm}_L{lyr}"]["mean_shift"] for lyr in LAYERS]
        ses = [summary[f"{arm}_L{lyr}"]["se"] for lyr in LAYERS]
        ax.errorbar(
            LAYERS,
            means,
            yerr=np.maximum(0, ses),
            fmt="-o",
            color=ARM_COLOR[arm],
            capsize=3,
            label=f"{arm} arm",
        )
    ax.axhline(0.0, ls=":", color="gray")
    ax.set_xticks(LAYERS)
    ax.set_xlabel("Steer layer (α = 4)")
    ax.set_ylabel("Graded judge shift vs baseline (points)")
    ax.set_title("Behavior: judge shift peaks at layer 14")
    ax.legend(fontsize=9)

    # --- Panel C: per-pair shifts (both arms), symlog ---
    ax = axes[2]
    rng = np.random.default_rng(42)
    idx = {lyr: i for i, lyr in enumerate(LAYERS)}
    for arm, dx in (("prefix", -0.18), ("context", 0.18)):
        for i, lyr in enumerate(LAYERS):
            sh = shifts[(arm, lyr)]
            xs = np.full(len(pairs), idx[lyr] + dx) + rng.uniform(-0.08, 0.08, len(pairs))
            ys = [sh[p] for p in pairs]
            ax.scatter(
                xs,
                ys,
                s=14,
                alpha=0.5,
                color=ARM_COLOR[arm],
                edgecolors="none",
                label=f"{arm} arm (per pair)" if i == 0 else None,
            )
    ax.text(
        idx[14] + 0.32, shifts[("context", 14)]["m685_04_terse"], "terse", fontsize=8, va="center"
    )
    ax.text(
        idx[17] + 0.32, shifts[("context", 17)]["m685_05_formal"], "formal", fontsize=8, va="center"
    )
    ax.set_yscale("symlog", linthresh=2)
    ax.axhline(0.0, ls=":", color="gray")
    ax.set_xticks(list(idx.values()), [str(lyr) for lyr in LAYERS])
    ax.set_xlabel("Steer layer (α = 4)")
    ax.set_ylabel("Per-pair judge shift (points, symlog)")
    ax.set_title("Per-pair shifts (28 pairs × 2 arms)")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    savefig_paper(fig, "layer_profile_geometry_vs_behavior", dir=FIGDIR)
    print(f"saved figure to {FIGDIR}")


if __name__ == "__main__":
    main()
