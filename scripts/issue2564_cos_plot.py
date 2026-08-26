"""#2564 — direction cosine cos(predicted Δ, observed Δ) per change type.

Companion to issue2564_shift_bars.py, using the SAME category set (fired
value-swap axes + query_content as "question topic") so the two figures read
consistently in the paper. Answers: does the frozen #779 context->answer map
predict the DIRECTION of the answer-vector change between two minimal-pair
contexts? 1 = exact direction, 0 = no directional info (dashed reference),
<0 = wrong way.

Instruction axes use the fired value-swap pairs (pair_class=swap, pair_fired_70);
the query axis is query_content ("question topic"). Point = median cos; error
bar = pair-level bootstrap 95% interval (B=10,000, seed 2564). Read-only; no GPU.

Input:  eval_results/issue_2564/perpair.jsonl (override with --perpair).
Output: figures/issue_2564/cos_direction_by_change_type.{png,pdf} (+ meta sidecar).
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


def _cos_vals(rows: list[dict]) -> np.ndarray:
    out = []
    for r in rows:
        v = (r.get("cos") or {}).get(MAP_ARM)
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
    by_axis: dict[str, list[dict]] = {}
    for r in fired:
        by_axis.setdefault(r["axis"], []).append(r)
    cts = [(axis.replace("_", " "), rr) for axis, rr in by_axis.items()]
    for cls, label in (("query_content", "question topic"),):
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

    rng = np.random.default_rng(SEED)
    recs = []
    for label, rr in cts:
        med, lo, hi = _boot_ci(_cos_vals(rr), rng)
        recs.append({"label": label, "n": len(_cos_vals(rr)), "val": med, "lo": lo, "hi": hi})
    recs.sort(key=lambda d: d["val"])

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    n = len(recs)
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * n + 1.6))
    col = paper_color("neural_map")
    for i, d in enumerate(recs):
        lo = min(d["lo"], d["val"])
        hi = max(d["hi"], d["val"])
        ax.plot([lo, hi], [i, i], color=col, lw=2.4, zorder=2, solid_capstyle="round")
        ax.scatter([d["val"]], [i], color=col, s=46, zorder=3)
    ax.axvline(0.0, color="0.4", lw=1.1, ls="--", zorder=1, label="no directional info")
    ax.set_yticks(range(n))
    ax.set_yticklabels([d["label"] for d in recs], fontsize=9)
    ax.set_xlabel(
        "direction  cos(predicted Δ, observed Δ)   (1 = exact direction, 0 = no directional info)"
    )
    ax.set_title("Does the map predict the DIRECTION of the answer-vector change? (#2564)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    paths = savefig_paper(fig, "cos_direction_by_change_type", dir=FIG_DIR)
    plt.close(fig)
    for d in recs:
        print(f"{d['label']:22s} n={d['n']:>4} cos={d['val']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]")
    print("wrote", paths.get("png"))


if __name__ == "__main__":
    main()
