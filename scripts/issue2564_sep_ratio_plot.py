"""Plot the separation ratio ‖predicted Δ‖ / ‖observed Δ‖ per change type for #2564.

Answers: for two minimal-pair contexts differing in one axis, does the frozen
context→answer map INFLATE (>1) or SUPPRESS (<1) the separation between their
answer representations, relative to the real answer separation (=1)? Both
predicted and observed deltas live in the same answer-representation space, so
the ratio is a direct same-units read (no carrier yardstick, unlike #2215's
sepcmp round).

Input: eval_results/issue_2564/perpair.jsonl (per-pair `norm_pred` {arm:...} +
`norm_obs_tail_L19`); path overridable with --perpair. Read-only.
Output: figures/issue_2564/sep_ratio_by_change_type.{png,pdf} (+ meta sidecar).

Instruction axes use the fired value-swap pairs (pair_class=swap, pair_fired_70);
query axes are their own pair classes (query_content / query_form /
query_paraphrase). Bars are the median ratio; error bars are a pair-level
bootstrap 95% interval (B=10,000, seed 2564).
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
ID_ARM = "arm_iddelta"
B = 10_000
SEED = 2564


def _ratios(rows: list[dict], arm: str) -> np.ndarray:
    out = []
    for r in rows:
        obs = r.get("norm_obs_tail_L19")
        pred = (r.get("norm_pred") or {}).get(arm)
        if obs and pred and obs > 0:
            out.append(pred / obs)
    return np.asarray(out, dtype=float)


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    meds = np.median(vals[idx], axis=1)
    return float(np.median(vals)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perpair", default=str(DEFAULT_PERPAIR))
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.perpair, encoding="utf-8")]
    swap = [r for r in rows if r["pair_class"] == "swap"]
    fired = [r for r in swap if r.get("pair_fired_70")] or swap

    # build the ordered change-type list
    change_types: list[tuple[str, list[dict]]] = []
    by_axis: dict[str, list[dict]] = {}
    for r in fired:
        by_axis.setdefault(r["axis"], []).append(r)
    for axis, rr in by_axis.items():
        change_types.append((axis.replace("_", " "), rr))
    for cls, label in (("query_content", "query content"),):
        rr = [r for r in rows if r["pair_class"] == cls]
        if rr:
            change_types.append((label, rr))

    rng = np.random.default_rng(SEED)
    recs = []
    for label, rr in change_types:
        mv = _ratios(rr, MAP_ARM)
        med, lo, hi = _boot_ci(mv, rng)
        recs.append({"label": label, "n": len(mv), "map": med, "lo": lo, "hi": hi})
    recs.sort(key=lambda d: d["map"])

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    n = len(recs)
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * n + 1.6))
    c_inf = paper_color("neural_map")
    c_sup = paper_color("oracle_answer")
    for i, d in enumerate(recs):
        col = c_inf if d["map"] > 1.0 else c_sup
        ax.plot([d["lo"], d["hi"]], [i, i], color=col, lw=2.4, zorder=2, solid_capstyle="round")
        ax.scatter([d["map"]], [i], color=col, s=46, zorder=3)
    ax.axvline(1.0, color="0.35", lw=1.1, ls="--", zorder=1)
    lo_lim = min(0.9, min(d["lo"] for d in recs) - 0.05)
    hi_lim = max(1.1, max(d["hi"] for d in recs) + 0.05)
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_yticks(range(n))
    ax.set_yticklabels([d["label"] for d in recs], fontsize=8)
    ax.set_xlabel(
        "separation ratio  ‖predicted Δ‖ / ‖observed Δ‖   (1 = matches real answer separation)"
    )
    ax.set_title(
        "Does the map inflate or suppress the separation between two similar contexts? (#2564)"
    )
    ax.scatter([], [], color=c_inf, s=46, label="map inflates (>1)")
    ax.scatter([], [], color=c_sup, s=46, label="map suppresses (<1)")
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    paths = savefig_paper(fig, "sep_ratio_by_change_type", dir=FIG_DIR)
    plt.close(fig)
    for d in recs:
        print(f"{d['label']:22s} n={d['n']:>4} map={d['map']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]")
    print("wrote", paths.get("png"))


if __name__ == "__main__":
    main()
