"""Issue #2094 user-chat inline round: steering-strength (dose) response lineplot.

Companion to ``scripts/issue2094_userchat_heatmaps.py`` (whose loaders/labels
this reuses). The heatmaps fix the dose at ``replace`` (full-state patch) and
sweep slot x layer; this script does the orthogonal cut — fix the best
slot/layer cell and sweep the STEERING STRENGTH alpha in {0.5, 1, 2, 4}, with
the full-state ``replace`` patch shown as a separate right-hand tick.

Three series on one panel (the write-up's Result 4):

- F_beh at the best-behavior cell (judged behavior fraction-of-swap)
- F_act at the best-activation cell (answer-vector fraction-of-swap)
- coherent-draw fraction (mean over the two selected cells)

Each F series is drawn against its own norm-matched shuffled-donor null
(dashed, same color). All means are over WELL-SEPARATED pairs only
(|anchor separation| >= --min-sep, the FA-3 convention of
``issue2094_wellsep_bootstrap.py``), coherent non-degenerate rows only.

Cell selection is stated in the caption and is deliberately dose-BLIND: the
cell is chosen by the mean of its metric ACROSS the four steered doses, so the
plotted dose curve is not selected on its own peak. Selection ranges over
every (setting, slot, layer_variant) cell that ran at all four doses.

Usage:
  uv run python scripts/issue2094_dose_lineplot.py \
      --eval-root eval_results/issue_2094 \
      --out-dir figures/issue_2094/userchat_heatmaps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue2094_userchat_heatmaps import (  # noqa: E402  (reuse, do not re-implement)
    LAYER_ROWS,
    SETTING_TITLES,
    SLOT_LABELS,
    beh_value,
)

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logger = logging.getLogger("issue2094_dose_lineplot")

DOSES = ("a0.5", "a1", "a2", "a4")
DOSE_ALPHA = {"a0.5": 0.5, "a1": 1.0, "a2": 2.0, "a4": 4.0}
REPLACE = "replace"


def load_rows(eval_root: Path, name: str) -> list[dict]:
    path = eval_root / "f_metrics" / name
    assert path.exists(), f"missing table: {path}"
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_wellsep(eval_root: Path, min_sep: float) -> tuple[set[tuple[str, str]], set[str]]:
    """(pair_id, rubric-kind) and pair-only sets at |separation| >= min_sep."""
    pair_kind: set[tuple[str, str]] = set()
    with (eval_root / "f_metrics" / "anchors.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            a = json.loads(line)
            sep = a.get("separation")
            if sep is not None and abs(sep) >= min_sep:
                pair_kind.add((a["pair_id"], a["kind"]))
    assert pair_kind, f"no well-separated anchors at |sep| >= {min_sep}"
    return pair_kind, {pid for pid, _ in pair_kind}


def collect(
    rows: list[dict], ws_pairs: set[str]
) -> dict[tuple[str, str, str, str], dict[str, list[float]]]:
    """-> {(setting, slot, layer_variant, dose): {f_act: [...], f_beh: [...], coh: [...]}}"""
    out: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        if r.get("degenerate_self") or r["layer_variant"] not in LAYER_ROWS:
            continue
        if r["dose"] not in (*DOSES, REPLACE):
            continue
        if r["pair_id"] not in ws_pairs:
            continue
        key = (r["setting"], r["slot"], r["layer_variant"], r["dose"])
        bucket = out[key]
        bucket["coh"].append(1.0 if r["coherent"] else 0.0)
        if not r["coherent"]:
            continue
        if r.get("f_act") is not None and not r.get("f_act_degenerate"):
            bucket["f_act"].append(float(r["f_act"]))
        bv = beh_value(r, r["setting"])
        if bv is not None:
            bucket["f_beh"].append(bv)
    return out


def pick_cell(
    steered: dict[tuple[str, str, str, str], dict[str, list[float]]],
    nulls: dict[tuple[str, str, str, str], dict[str, list[float]]],
    metric: str,
    min_pairs: int,
    min_coh: float,
    single_layer_only: bool = False,
) -> tuple[str, str, str]:
    """Dose-blind argmax of the MARGIN over the shuffled-donor null.

    Ranking on the raw steered mean selects the span-slot cells whose null is
    just as large (F up to ~6 from cap-hit-truncated rollouts against a
    near-floor denominator) — those are artifacts, not causal effects. The
    effect size is steered - null, so that is what selection maximizes.
    Cells whose coherent-draw fraction falls below ``min_coh`` at any dose are
    ineligible (the same <50%-coherent cells the heatmaps flag with '*').
    """
    scores: dict[tuple[str, str, str], float] = {}
    for cell in {k[:3] for k in steered}:
        if single_layer_only and not cell[2].startswith("L"):
            continue
        per_dose = []
        for dose in DOSES:
            s = steered.get((*cell, dose), {})
            n = nulls.get((*cell, dose), {}).get(metric, [])
            s_vals, coh = s.get(metric, []), s.get("coh", [])
            if len(s_vals) < min_pairs or len(n) < min_pairs:
                per_dose = []
                break
            if not coh or float(np.mean(coh)) < min_coh:
                per_dose = []
                break
            per_dose.append(float(np.mean(s_vals)) - float(np.mean(n)))
        if len(per_dose) == len(DOSES):
            scores[cell] = float(np.mean(per_dose))
    assert scores, f"no eligible cell at all doses (>= {min_pairs} pairs, coh >= {min_coh})"
    return max(scores, key=lambda c: scores[c])


def series(
    table: dict[tuple[str, str, str, str], dict[str, list[float]]],
    cell: tuple[str, str, str],
    metric: str,
) -> tuple[list[float], list[int], float | None, int]:
    means, ns = [], []
    for dose in DOSES:
        vals = table.get((*cell, dose), {}).get(metric, [])
        means.append(float(np.mean(vals)) if vals else np.nan)
        ns.append(len(vals))
    rep = table.get((*cell, REPLACE), {}).get(metric, [])
    return means, ns, (float(np.mean(rep)) if rep else None), len(rep)


def label(cell: tuple[str, str, str]) -> str:
    setting, slot, layer = cell
    return (
        f"{SETTING_TITLES[setting].split(' (')[0]}, "
        f"{SLOT_LABELS[slot].replace(chr(10), ' ')} {layer}"
    )


def pick_best_single_layer(steered, nulls, metric, min_pairs, min_coh, setting, slot):
    """Dose-blind argmax of the null-margin among SINGLE layers at one (setting, slot)."""
    scores = {}
    for cell in {k[:3] for k in steered}:
        if cell[0] != setting or cell[1] != slot or not cell[2].startswith("L"):
            continue
        per_dose = []
        for dose in DOSES:
            sv = steered.get((*cell, dose), {})
            nv = nulls.get((*cell, dose), {}).get(metric, [])
            vals, coh = sv.get(metric, []), sv.get("coh", [])
            if len(vals) < min_pairs or len(nv) < min_pairs:
                per_dose = []
                break
            if not coh or float(np.mean(coh)) < min_coh:
                per_dose = []
                break
            per_dose.append(float(np.mean(vals)) - float(np.mean(nv)))
        if len(per_dose) == len(DOSES):
            scores[cell] = float(np.mean(per_dose))
    assert scores, f"no eligible single-layer {slot} cell at {setting}"
    return max(scores, key=lambda c: scores[c])[2]


def figure_by_layer_variant(steered, nulls, args, setting="matched_query", slot="ce") -> dict:
    """Result 4: dose response at the CONTEXT VECTOR, one panel per layer variant.

    The companion figure fixes the best cell and sweeps alpha; this one fixes the
    slot to the context vector and asks the same dose question separately at the
    best single layer, the middle band, and all 28 layers -- the three depths the
    write-up's Result 4 names. Three series per panel: judged behavior transfer,
    answer-vector transfer, and the coherent-draw fraction, each F series drawn
    against its own norm-matched shuffled-donor null.
    """
    best_L = pick_best_single_layer(
        steered, nulls, "f_beh", args.min_pairs, args.min_coh, setting, slot
    )
    variants = [
        (best_L, f"best single layer ({best_L})"),
        ("joint_mid", "middle band (layers 14-20)"),
        ("joint_all", "all 28 layers"),
    ]
    colors = paper_palette(3)
    x = np.log2([DOSE_ALPHA[d] for d in DOSES])
    x_rep = x[-1] + 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True)
    out: dict = {}
    for ax, (lv, title) in zip(axes, variants):
        cell = (setting, slot, lv)
        for (metric, mlabel), color in zip(
            (("f_beh", "behavior transfer (F_beh)"), ("f_act", "answer-vector transfer (F_act)")),
            colors,
        ):
            m, n, rep, n_rep = series(steered, cell, metric)
            nm, _nn, nrep, _ = series(nulls, cell, metric)
            ax.plot(x, m, marker="o", color=color, label=f"{mlabel} (n={max(n) if n else 0})")
            ax.plot(
                x,
                nm,
                marker="",
                ls="--",
                lw=1.2,
                color=color,
                alpha=0.75,
                label=f"{mlabel} — donor null",
            )
            if rep is not None:
                ax.plot([x_rep], [rep], marker="*", ms=13, color=color, ls="none")
            if nrep is not None:
                ax.plot([x_rep], [nrep], marker="*", ms=9, mfc="none", color=color, ls="none")
            out[f"{lv}|{metric}"] = {"alpha": m, "null": nm, "replace": rep, "n": n}
        cm, cn, crep, _ = series(steered, cell, "coh")
        ax.plot(
            x,
            cm,
            marker="^",
            color=colors[2],
            label=f"coherent-draw fraction (n={max(cn) if cn else 0})",
        )
        if crep is not None:
            ax.plot([x_rep], [crep], marker="*", ms=13, color=colors[2], ls="none")
        out[f"{lv}|coh"] = {"alpha": cm, "replace": crep, "n": cn}
        ax.axhline(0.0, color="0.45", lw=0.9)
        ax.set_xticks([*x, x_rep])
        ax.set_xticklabels([*(f"{DOSE_ALPHA[d]:g}" for d in DOSES), "replace\n(full state)"])
        ax.set_xlabel("steering strength alpha  (log spaced)")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("fraction of a full context swap")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Result 4: steering at the context vector — dose response by depth "
        f"({SETTING_TITLES[setting].split(' (')[0]}, well-separated pairs)\n"
        "solid = real steering, dashed = norm-matched shuffled-donor null, star = full-state patch; "
        "alpha*Delta with Delta = v_C(B) - v_C(A)",
        fontsize=10.5,
    )
    fig.tight_layout()
    savefig_paper(fig, "dose_lineplot_by_layer", dir=args.out_dir)
    plt.close(fig)
    out["best_single_layer"] = best_L
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", type=Path, default=Path("eval_results/issue_2094"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures/issue_2094/userchat_heatmaps"))
    ap.add_argument("--min-sep", type=float, default=0.5)
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--min-coh", type=float, default=0.5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _, ws_pairs = load_wellsep(args.eval_root, args.min_sep)
    steered = collect(
        [r for r in load_rows(args.eval_root, "f_cells.jsonl") if r["arm"] == "steered"], ws_pairs
    )
    nulls = collect(
        [r for r in load_rows(args.eval_root, "null_cells.jsonl") if r["arm"] == "null"], ws_pairs
    )

    set_paper_style()
    colors = paper_palette(3)
    x = np.log2([DOSE_ALPHA[d] for d in DOSES])
    x_rep = x[-1] + 1.0
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    panels: dict[str, dict] = {}

    for ax, single_only, panel_title in (
        (axes[0], False, "best cell overall (joint edits eligible)"),
        (axes[1], True, "best SINGLE-layer cell (α = 1 ≡ patching that layer)"),
    ):
        beh_cell = pick_cell(steered, nulls, "f_beh", args.min_pairs, args.min_coh, single_only)
        act_cell = pick_cell(steered, nulls, "f_act", args.min_pairs, args.min_coh, single_only)
        logger.info(
            "[dose-lineplot] single_layer_only=%s behavior=%s activation=%s",
            single_only,
            beh_cell,
            act_cell,
        )
        beh, beh_n, beh_rep, beh_rep_n = series(steered, beh_cell, "f_beh")
        beh_null, _, beh_null_rep, _ = series(nulls, beh_cell, "f_beh")
        act, act_n, act_rep, act_rep_n = series(steered, act_cell, "f_act")
        act_null, _, act_null_rep, _ = series(nulls, act_cell, "f_act")
        coh_b, _, coh_b_rep, _ = series(steered, beh_cell, "coh")
        coh_a, _, coh_a_rep, _ = series(steered, act_cell, "coh")
        coh = [float(np.nanmean([b, a])) for b, a in zip(coh_b, coh_a)]
        reps = [v for v in (coh_b_rep, coh_a_rep) if v is not None]
        coh_rep = float(np.nanmean(reps)) if reps else None

        for vals, null_vals, rep, null_rep, color, name in (
            (beh, beh_null, beh_rep, beh_null_rep, colors[0], "behavior F"),
            (act, act_null, act_rep, act_null_rep, colors[1], "answer-vector F"),
        ):
            ax.plot(x, vals, "o-", color=color, label=name)
            ax.plot(x, null_vals, "s--", color=color, alpha=0.55, label=f"{name} — null")
            if rep is not None:
                ax.plot([x_rep], [rep], "o", color=color)
            if null_rep is not None:
                ax.plot([x_rep], [null_rep], "s", color=color, alpha=0.55)
        ax.plot(x, coh, "^-", color=colors[2], label="coherent-draw fraction")
        if coh_rep is not None:
            ax.plot([x_rep], [coh_rep], "^", color=colors[2])

        ax.axhline(0.0, color="0.6", lw=0.8, zorder=0)
        ax.axvline((x[-1] + x_rep) / 2, color="0.8", lw=0.8, ls=":", zorder=0)
        ax.set_xticks([*x, x_rep])
        ax.set_xticklabels([*(f"{DOSE_ALPHA[d]:g}" for d in DOSES), "replace\n(full state)"])
        ax.set_xlabel("steering strength α")
        ax.set_title(
            f"{panel_title}\nbehavior: {label(beh_cell)} | answer-vector: {label(act_cell)}",
            fontsize=8,
        )
        panels["overall" if not single_only else "single_layer"] = {
            "behavior_cell": dict(zip(("setting", "slot", "layer_variant"), beh_cell)),
            "activation_cell": dict(zip(("setting", "slot", "layer_variant"), act_cell)),
            "f_beh": beh,
            "f_beh_null": beh_null,
            "f_beh_replace": beh_rep,
            "f_beh_replace_null": beh_null_rep,
            "f_act": act,
            "f_act_null": act_null,
            "f_act_replace": act_rep,
            "f_act_replace_null": act_null_rep,
            "coherent_fraction": coh,
            "coherent_fraction_replace": coh_rep,
            "n_pairs_per_dose": {"behavior": beh_n, "activation": act_n},
            "n_pairs_replace": {"behavior": beh_rep_n, "activation": act_rep_n},
        }

    axes[0].set_ylabel("metric value")
    axes[0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle(
        "Steering-strength response — steered (solid) vs shuffled-donor null (dashed)",
        fontsize=10,
    )
    fig.text(
        0.005,
        0.005,
        f"well-separated pairs only (|anchor sep| ≥ {args.min_sep}); coherent non-degenerate rows; "
        f"cells chosen dose-blind (argmax of the mean steered−null MARGIN over "
        f"α ∈ {{0.5,1,2,4}}; ≥{args.min_pairs} pairs and ≥{args.min_coh:.0%} coherent at every dose)",
        fontsize=6,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # savefig_paper takes a STEM + dir (it appends .png/.pdf/.meta.json itself).
    savefig_paper(fig, "dose_lineplot", dir=args.out_dir)
    plt.close(fig)

    summary = {
        "panels": panels,
        "doses": list(DOSES),
        "min_abs_separation": args.min_sep,
        "min_pairs": args.min_pairs,
        "min_coherent_fraction": args.min_coh,
        "selection_rule": (
            "dose-blind argmax over (setting, slot, layer_variant) of the mean steered-null "
            "margin across the four steered doses; well-separated pairs, coherent "
            "non-degenerate rows, cells >= min_coh coherent at every dose"
        ),
    }
    summary["by_layer_variant"] = figure_by_layer_variant(steered, nulls, args)
    (args.out_dir / "dose_lineplot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("[phase=dose_lineplot_done] -> %s", args.out_dir / "dose_lineplot.png")


if __name__ == "__main__":
    main()
