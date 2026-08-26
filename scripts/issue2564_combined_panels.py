"""#2564 — the four per-change-type reads as one side-by-side panel figure.

Four panels sharing the change-type (y) axis, so each change type reads across
all four metrics on one row:

  1. separation ratio  ‖predicted Δ‖ / ‖observed Δ‖  (>1 inflate, <1 suppress; ref 1.0)
  2. direction         cos(predicted Δ, observed Δ)   (1 = exact direction; ref 0)
  3. real shift        ‖observed Δ‖                    (real answer-vector move; ref = noise floor)
  4. mapped shift      ‖predicted Δ‖                   (map's predicted move; same x-scale as panel 3)

Frozen #779 single-turn map (arm_779ce). Instruction axes use fired value-swap
pairs (pair_class=swap, pair_fired_70); the query axis is query_content. Rows
ordered by real-shift median (ascending). Point = median; bar = pair-level
bootstrap 95% interval (B=10,000, seed 2564). Read-only; no GPU.

Input:  eval_results/issue_2564/perpair.jsonl (override with --perpair).
Output: figures/issue_2564/all_reads_by_change_type.{png,pdf} (+ meta sidecar).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # before numpy — thread-pool freezes from OMP_NUM_THREADS at import

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PERPAIR = REPO / "eval_results/issue_2564/perpair.jsonl"
FIG_DIR = REPO / "figures/issue_2564"
MAP_ARM = "arm_779ce"
B = 10_000
SEED = 2564


def _vals(rows: list[dict], metric: str) -> np.ndarray:
    out = []
    for r in rows:
        if metric == "ratio":
            obs = r.get("norm_obs_tail_L19")
            pred = (r.get("norm_pred") or {}).get(MAP_ARM)
            v = (pred / obs) if (obs and pred and obs > 0) else None
        elif metric == "cos":
            v = (r.get("cos") or {}).get(MAP_ARM)
        elif metric == "real":
            v = r.get("norm_obs_tail_L19")
        elif metric == "pred":
            v = (r.get("norm_pred") or {}).get(MAP_ARM)
        elif metric == "ctx":
            # identity-baseline delta = v_C(A) - v_C(B): the map passes the
            # context-end vector straight through (bias cancels in the delta),
            # so its predicted "answer" delta IS the raw context-vector shift.
            v = (r.get("norm_pred") or {}).get("arm_iddelta")
        elif metric == "noise":
            v = r.get("noise_norm")
        else:
            raise ValueError(metric)
        if v is not None:
            out.append(float(v))
    return np.asarray(out, dtype=float)


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    meds = np.median(vals[idx], axis=1)
    return float(np.median(vals)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def _change_types(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Every varied change type. Instruction axes use fired value-swap pairs
    where any fired; axes with ZERO fired pairs (manipulation check never
    tripped) fall back to all their swap pairs, tagged '(not fired)'. Query
    axes (content / form / paraphrase) have no fire gate and use all pairs."""
    swap = [r for r in rows if r["pair_class"] == "swap"]
    by_axis: dict[str, list[dict]] = {}
    for r in swap:
        by_axis.setdefault(r["axis"], []).append(r)
    cts: list[tuple[str, list[dict]]] = []
    for axis, rr in by_axis.items():
        fired = [r for r in rr if r.get("pair_fired_70")]
        label = axis.replace("_", " ") + ("" if fired else " (not fired)")
        cts.append((label, fired if fired else rr))
    for cls, label in (
        ("query_content", "query content"),
        ("query_form", "query form"),
        ("query_paraphrase", "query paraphrase"),
    ):
        rr = [r for r in rows if r["pair_class"] == cls]
        if rr:
            cts.append((label, rr))
    return cts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perpair", default=str(DEFAULT_PERPAIR))
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.perpair, encoding="utf-8")]
    cts = _change_types(rows)
    noise_ref = float(np.median(_vals([r for r in rows if r["pair_class"] == "swap"], "noise")))

    rng = np.random.default_rng(SEED)
    # median+CI per (change type, metric)
    stats: dict[str, dict[str, tuple[float, float, float]]] = {}
    for label, rr in cts:
        stats[label] = {
            m: _boot_ci(_vals(rr, m), rng) for m in ("ratio", "cos", "real", "pred", "ctx")
        }
    order = sorted(stats, key=lambda label: stats[label]["real"][0])  # ascending real shift

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    c_inf, c_sup = paper_color("neural_map"), paper_color("oracle_answer")
    c_map, c_real = paper_color("neural_map"), paper_color("oracle_answer")
    n = len(order)
    fig, axes = plt.subplots(1, 5, figsize=(22.0, 0.5 * n + 2.5), sharey=True)

    def draw(ax, metric, color_fn, ref, xlabel, title):
        for i, label in enumerate(order):
            med, lo, hi = stats[label][metric]
            c = color_fn(med)
            ax.plot([lo, hi], [i, i], color=c, lw=2.4, zorder=2, solid_capstyle="round")
            ax.scatter([med], [i], color=c, s=42, zorder=3)
        if ref is not None:
            ax.axvline(ref, color="0.4", lw=1.1, ls="--", zorder=1)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)

    c_ctx = "0.45"
    draw(
        axes[0],
        "ctx",
        lambda v: c_ctx,
        None,
        "‖v_C(A) − v_C(B)‖ (L19 units)",
        "context vector shift\n(the map's input)",
    )
    draw(
        axes[1],
        "real",
        lambda v: c_real,
        noise_ref,
        "‖obs Δ‖ (L19 units)",
        "real answer shift\n(dashed = noise floor)",
    )
    draw(
        axes[2],
        "pred",
        lambda v: c_map,
        None,
        "‖pred Δ‖ (L19 units)",
        "mapped answer shift\n(map's prediction)",
    )
    draw(
        axes[3],
        "ratio",
        lambda v: c_inf if v > 1.0 else c_sup,
        1.0,
        "‖pred Δ‖ / ‖obs Δ‖",
        "separation ratio\n(>1 inflate, <1 suppress)",
    )
    draw(
        axes[4],
        "cos",
        lambda v: c_map,
        0.0,
        "cos(pred Δ, obs Δ)",
        "direction\n(1 = exact, 0 = chance)",
    )

    # real (panel 1) & mapped (panel 2) share the residual-stream x-scale for
    # direct visual comparison; context shift (panel 0) is a bigger regime, own scale
    lo1 = min(stats[label]["real"][1] for label in order)
    lo2 = min(stats[label]["pred"][1] for label in order)
    hi1 = max(stats[label]["real"][2] for label in order)
    hi2 = max(stats[label]["pred"][2] for label in order)
    shared = (min(lo1, lo2, noise_ref) - 2, max(hi1, hi2) + 2)
    axes[1].set_xlim(*shared)
    axes[2].set_xlim(*shared)

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(order, fontsize=9)
    fig.suptitle("Context→answer map per change type (#2564)", fontsize=12, y=1.02)
    fig.tight_layout()
    paths = savefig_paper(fig, "all_reads_by_change_type", dir=FIG_DIR)
    plt.close(fig)
    for label in reversed(order):
        s = stats[label]
        print(
            f"{label:24s} ctx={s['ctx'][0]:.1f} real={s['real'][0]:.1f} pred={s['pred'][0]:.1f} "
            f"ratio={s['ratio'][0]:.2f} cos={s['cos'][0]:.2f}"
        )
    print(f"noise floor = {noise_ref:.2f}")
    print("wrote", paths.get("png"))


if __name__ == "__main__":
    main()
