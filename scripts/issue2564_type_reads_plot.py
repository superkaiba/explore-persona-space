"""Per-change-type reads for #2564: direction cosine and real answer-vector shift.

Companion to `issue2564_sep_ratio_plot.py` (which plots the magnitude RATIO
‖predicted Δ‖/‖observed Δ‖). This script renders the two component reads as
SEPARATE standalone figures, per change type:

  --metric cos    : median cos(Δ̂, Δ) for the frozen #779 single-turn map
                    (arm_779ce) — does the map predict the DIRECTION of the
                    answer-vector change between the two contexts? 1 = exact,
                    0 = no directional info (dashed reference), <0 = wrong way.
  --metric real   : median ‖observed Δ‖ (norm_obs_tail_L19) — how much the REAL
                    answer representation actually moves between the two
                    contexts (map-independent). Dashed reference = median
                    per-pair noise floor (split-half noise magnitude).

Instruction axes use the fired value-swap pairs (pair_class=swap, pair_fired_70);
the query axis is query_content. Point = median; bar = pair-level bootstrap 95%
interval (B=10,000, seed 2564). Read-only; no model calls, no GPU.

Input:  eval_results/issue_2564/perpair.jsonl (override with --perpair).
Output: figures/issue_2564/<name>_by_change_type.{png,pdf} (+ meta sidecar).
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
        if metric == "cos":
            v = (r.get("cos") or {}).get(MAP_ARM)
        elif metric == "real":
            v = r.get("norm_obs_tail_L19")
        elif metric == "pred":
            v = (r.get("norm_pred") or {}).get(MAP_ARM)
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
    swap = [r for r in rows if r["pair_class"] == "swap"]
    fired = [r for r in swap if r.get("pair_fired_70")] or swap
    change_types: list[tuple[str, list[dict]]] = []
    by_axis: dict[str, list[dict]] = {}
    for r in fired:
        by_axis.setdefault(r["axis"], []).append(r)
    for axis, rr in by_axis.items():
        change_types.append((axis.replace("_", " "), rr))
    qc = [r for r in rows if r["pair_class"] == "query_content"]
    if qc:
        change_types.append(("query content", qc))
    return change_types


CONFIG = {
    "cos": {
        "name": "cos_by_change_type",
        "color": "neural_map",
        "ref": 0.0,
        "ref_label": "no directional info",
        "xlabel": "direction  cos(predicted Δ, observed Δ)   (1 = map predicts the exact shift direction)",
        "title": "Does the map predict the DIRECTION of the answer-vector change? (#2564)",
    },
    "real": {
        "name": "real_answer_shift_by_change_type",
        "color": "oracle_answer",
        "ref": None,  # noise floor, computed per run
        "ref_label": "median noise floor",
        "xlabel": "real answer-vector shift  ‖observed Δ‖  (layer-19 residual-stream units)",
        "title": "How much does the REAL answer vector move between the two contexts? (#2564)",
    },
    "pred": {
        "name": "mapped_answer_shift_by_change_type",
        "color": "neural_map",
        "ref": None,
        "ref_label": "",
        "xlabel": "mapped answer-vector shift  ‖predicted Δ‖  (layer-19 residual-stream units)",
        "title": "How much answer-vector shift does the MAP predict? (#2564)",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perpair", default=str(DEFAULT_PERPAIR))
    ap.add_argument("--metric", choices=("cos", "real", "pred"), required=True)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.perpair, encoding="utf-8")]
    change_types = _change_types(rows)
    cfg = CONFIG[args.metric]

    rng = np.random.default_rng(SEED)
    recs = []
    for label, rr in change_types:
        vals = _vals(rr, args.metric)
        med, lo, hi = _boot_ci(vals, rng)
        recs.append({"label": label, "n": len(vals), "val": med, "lo": lo, "hi": hi})
    recs.sort(key=lambda d: d["val"])

    ref = cfg["ref"]
    if args.metric == "real":
        ref = float(np.median(_vals([r for r in rows if r["pair_class"] == "swap"], "noise")))

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    n = len(recs)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * n + 1.6))
    col = paper_color(cfg["color"])
    for i, d in enumerate(recs):
        ax.plot([d["lo"], d["hi"]], [i, i], color=col, lw=2.4, zorder=2, solid_capstyle="round")
        ax.scatter([d["val"]], [i], color=col, s=46, zorder=3)
    if ref is not None:
        ax.axvline(ref, color="0.4", lw=1.1, ls="--", zorder=1, label=cfg["ref_label"])
        ax.legend(loc="lower right", fontsize=7.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([d["label"] for d in recs], fontsize=8)
    ax.set_xlabel(cfg["xlabel"])
    ax.set_title(cfg["title"])
    fig.tight_layout()
    paths = savefig_paper(fig, cfg["name"], dir=FIG_DIR)
    plt.close(fig)
    for d in recs:
        print(
            f"{d['label']:22s} n={d['n']:>4} {args.metric}={d['val']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]"
        )
    if ref is not None:
        print(f"reference ({cfg['ref_label']}) = {ref:.3f}")
    print("wrote", paths.get("png"))


if __name__ == "__main__":
    main()
