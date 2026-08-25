"""#2564 — real vs mapped answer-vector shift per change type, with the ratio below.

Two stacked panels sharing the change-type x-axis:
  TOP    grouped bars per category: real answer shift ‖observed Δ‖
         (norm_obs_tail_L19) vs mapped answer shift ‖predicted Δ‖
         (norm_pred[arm_779ce]) — how far the real answer vector moves between
         the two minimal-pair contexts vs how far the frozen #779 map predicts.
  BOTTOM the separation ratio ‖predicted Δ‖ / ‖observed Δ‖ per category
         (>1 map inflates, <1 map suppresses; dashed reference at 1.0).

Instruction axes use the fired value-swap pairs (pair_class=swap, pair_fired_70);
the query axis is query_content (labeled "question topic"). Point = median; error
bars = pair-level bootstrap 95% interval (B=10,000, seed 2564). Read-only; no GPU.

--extra-perpair appends the 2026-08-25 lang/oneword pilot categories
(eval_results/issue_2564/lang_oneword_pilot/perpair.jsonl): answer_language
fired swaps (label "answer language"; fired = programmatic language check) and
query_content_oneword (label "question topic (one word)"; no fired gate, same
as the parent query axis). Pilot rows carry the norm fields flattened
(norm_pred_arm_779ce) instead of the parent's norm_pred dict — both are read.

Input:  eval_results/issue_2564/perpair.jsonl (override with --perpair).
Output: figures/issue_2564/<out-stem>.{png,pdf} (+ meta sidecar);
        default stem shift_vs_ratio_bars (unchanged without the new flags).
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


def _pair_vals(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (real ‖obsΔ‖, mapped ‖predΔ‖, ratio) arrays over pairs with both."""
    real, pred, ratio = [], [], []
    for r in rows:
        obs = r.get("norm_obs_tail_L19")
        np_d = r.get("norm_pred")
        # parent perpair nests norms per arm; the lang/oneword pilot flattens them
        pr = np_d.get(MAP_ARM) if isinstance(np_d, dict) else r.get(f"norm_pred_{MAP_ARM}")
        if obs and pr and obs > 0:
            real.append(obs)
            pred.append(pr)
            ratio.append(pr / obs)
    return (
        np.asarray(real, dtype=float),
        np.asarray(pred, dtype=float),
        np.asarray(ratio, dtype=float),
    )


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


def _extra_change_types(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Pilot categories: answer-language fired swaps + one-word query switches."""
    cts: list[tuple[str, list[dict]]] = []
    lang = [
        r
        for r in rows
        if r["pair_class"] == "swap" and r["axis"] == "answer_language" and r.get("pair_fired_70")
    ]
    if lang:
        cts.append(("answer language", lang))
    oneword = [r for r in rows if r["pair_class"] == "query_content_oneword"]
    if oneword:
        cts.append(("question topic (one word)", oneword))
    return cts


def _errs(med: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Non-negative [lower, upper] offsets for matplotlib yerr (gotchas: clamp)."""
    return np.vstack([np.maximum(0.0, med - lo), np.maximum(0.0, hi - med)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perpair", default=str(DEFAULT_PERPAIR))
    ap.add_argument("--extra-perpair", default=None, help="pilot perpair.jsonl to append")
    ap.add_argument("--out-stem", default="shift_vs_ratio_bars")
    ap.add_argument("--exclude", default="", help="comma-separated tick labels to drop")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.perpair, encoding="utf-8")]
    cts = _change_types(rows)
    if args.extra_perpair:
        extra_rows = [json.loads(line) for line in open(args.extra_perpair, encoding="utf-8")]
        cts.extend(_extra_change_types(extra_rows))
    excl = {s.strip() for s in args.exclude.split(",") if s.strip()}
    if excl:
        unknown = excl - {label for label, _ in cts}
        assert not unknown, f"--exclude labels not in figure: {sorted(unknown)}"
        cts = [(label, rr) for label, rr in cts if label not in excl]

    rng = np.random.default_rng(SEED)
    recs = []
    for label, rr in cts:
        real, pred, ratio = _pair_vals(rr)
        rm, rl, rh = _boot_ci(real, rng)
        pm, pl, ph = _boot_ci(pred, rng)
        qm, ql, qh = _boot_ci(ratio, rng)
        recs.append(
            {
                "label": label,
                "n": len(real),
                "real": (rm, rl, rh),
                "pred": (pm, pl, ph),
                "ratio": (qm, ql, qh),
            }
        )
    recs.sort(key=lambda d: d["ratio"][0])  # ascending separation ratio (left to right)

    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_color,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    c_real = paper_color("oracle_answer")
    c_map = paper_color("neural_map")

    n = len(recs)
    x = np.arange(n)
    labels = [d["label"] for d in recs]
    real_m = np.array([d["real"][0] for d in recs])
    real_l = np.array([d["real"][1] for d in recs])
    real_h = np.array([d["real"][2] for d in recs])
    pred_m = np.array([d["pred"][0] for d in recs])
    pred_l = np.array([d["pred"][1] for d in recs])
    pred_h = np.array([d["pred"][2] for d in recs])
    rat_m = np.array([d["ratio"][0] for d in recs])
    rat_l = np.array([d["ratio"][1] for d in recs])
    rat_h = np.array([d["ratio"][2] for d in recs])

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(max(9.0, 1.15 * n + 2.0), 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )

    w = 0.38
    ax0.bar(
        x - w / 2,
        real_m,
        w,
        yerr=_errs(real_m, real_l, real_h),
        color=c_real,
        label="real answer shift  ‖observed Δ‖",
        capsize=3,
        error_kw={"lw": 1.0},
    )
    ax0.bar(
        x + w / 2,
        pred_m,
        w,
        yerr=_errs(pred_m, pred_l, pred_h),
        color=c_map,
        label="mapped answer shift  ‖predicted Δ‖",
        capsize=3,
        error_kw={"lw": 1.0},
    )
    ax0.set_ylabel("answer-vector shift  ‖Δ‖\n(layer-19 residual-stream units)")
    ax0.set_title("Real vs mapped answer-vector shift per change type (#2564)")
    ax0.legend(loc="upper left", fontsize=8)

    # neutral gray so orange/pink keep ONE meaning (mapped/real) across the whole
    # figure; the dashed 1.0 line shows inflate (>1) vs suppress (<1) by position.
    ax1.bar(
        x, rat_m, 0.6, yerr=_errs(rat_m, rat_l, rat_h), color="0.6", capsize=3, error_kw={"lw": 1.0}
    )
    ax1.axhline(1.0, color="0.35", lw=1.1, ls="--", zorder=1)
    ax1.set_ylabel("ratio\nmapped / real")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    fig.tight_layout()
    paths = savefig_paper(fig, args.out_stem, dir=FIG_DIR)
    plt.close(fig)
    for d in recs:
        print(
            f"{d['label']:22s} n={d['n']:>4} real={d['real'][0]:6.2f} "
            f"pred={d['pred'][0]:6.2f} ratio={d['ratio'][0]:.3f}"
        )
    print("wrote", paths.get("png"))


if __name__ == "__main__":
    main()
